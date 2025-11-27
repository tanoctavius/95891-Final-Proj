"""
Poshmark Listing Scraper Module
Scrapes item information from Poshmark listings with security measures
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import json
from typing import Dict, Optional, List
from PIL import Image
from io import BytesIO
import re
from urllib.parse import urlparse, urljoin, quote_plus

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    print("Warning: brotli not installed. Install with: pip install brotli")


class PoshmarkScraper:
    """Scraper for Poshmark listings with security and rate limiting"""
    
    # Rotating user agents to avoid detection
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self):
        """Initialize scraper with default settings"""
        self.session = requests.Session()
        self.min_delay = 2  # Minimum delay between requests (seconds)
        self.max_delay = 5  # Maximum delay between requests (seconds)
        self.timeout = 15  # Request timeout (seconds)
        
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string"""
        return random.choice(self.USER_AGENTS)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with random user agent and common browser headers"""
        return {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
    
    def _random_delay(self):
        """Add random delay between requests to avoid rate limiting"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
    
    def _validate_url(self, url: str) -> bool:
        """Validate that URL is a Poshmark listing URL"""
        if not url:
            return False
        
        parsed = urlparse(url)
        return (
            parsed.netloc in ['poshmark.com', 'www.poshmark.com'] and
            'listing' in parsed.path
        )
    
    def scrape_listing(self, url: str) -> Dict:
        """
        Scrape a Poshmark listing
        
        Args:
            url: Poshmark listing URL
            
        Returns:
            Dictionary containing:
                - title: Item title/name
                - size: Item size
                - description: Item description
                - images: List of image URLs
                - price: Item price (if available)
                - brand: Brand name (if available)
                - success: Whether scraping was successful
                - error: Error message if failed
        """
        result = {
            'success': False,
            'title': None,
            'size': None,
            'description': None,
            'images': [],
            'price': None,
            'brand': None,
            'error': None
        }
        
        # Validate URL
        if not self._validate_url(url):
            result['error'] = 'Invalid Poshmark URL. Please provide a valid Poshmark listing URL.'
            return result
        
        try:
            # Add delay before request
            self._random_delay()
            
            # Make request with headers
            headers = self._get_headers()
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            # Check if request was successful
            if response.status_code != 200:
                result['error'] = f'Failed to fetch page. Status code: {response.status_code}'
                return result
            
            # Use response.text which automatically handles decompression
            # If Brotli is available, requests will handle it automatically
            # Otherwise, try to decode manually
            try:
                html_content = response.text
            except UnicodeDecodeError:
                # Fallback: try to decompress manually if needed
                content_encoding = response.headers.get('Content-Encoding', '').lower()
                if content_encoding == 'br' and BROTLI_AVAILABLE:
                    try:
                        decompressed = brotli.decompress(response.content)
                        html_content = decompressed.decode('utf-8')
                    except Exception as e:
                        result['error'] = f'Failed to decompress/decode content: {str(e)}'
                        return result
                else:
                    html_content = response.content.decode('utf-8', errors='ignore')
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            result['title'] = title
            
            # Extract size
            size = self._extract_size(soup)
            result['size'] = size
            
            # Extract description
            description = self._extract_description(soup)
            result['description'] = description
            
            # Extract images
            images = self._extract_images(soup, url)
            result['images'] = images
            
            # Extract price (optional)
            price = self._extract_price(soup)
            result['price'] = price
            
            # Extract brand (optional)
            brand = self._extract_brand(soup)
            result['brand'] = brand
            
            # Mark as successful if we got at least title or description
            if title or description:
                result['success'] = True
            else:
                result['error'] = 'Could not extract item information from page'
                
        except requests.exceptions.Timeout:
            result['error'] = 'Request timed out. Please try again later.'
        except requests.exceptions.RequestException as e:
            result['error'] = f'Network error: {str(e)}'
        except Exception as e:
            result['error'] = f'Unexpected error: {str(e)}'
        
        return result
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract item title from the page"""
        # Try multiple selectors for title
        selectors = [
            'h1[data-testid="listing-title"]',
            'h1.listing-title',
            'h1.title',
            'h1',
            '[data-testid="title"]',
            '.title h1',
        ]
        
        for selector in selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title:
                    return title
        
        # Fallback: look for title in meta tags
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title.get('content').strip()
        
        return None
    
    def _extract_size(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract item size from the page"""
        # Try multiple selectors for size - prioritize data-testid
        selectors = [
            '[data-testid="size"]',
            '[class*="size"][class*="value"]',
            'span:contains("Size"):not(:contains("BOUTIQUES"))',
        ]
        
        for selector in selectors:
            size_elem = soup.select_one(selector)
            if size_elem:
                size_text = size_elem.get_text(strip=True)
                # Skip if it's clearly not a size (like BOUTIQUES, Categories, etc.)
                if any(skip in size_text.upper() for skip in ['BOUTIQUES', 'CATEGORY', 'CATEGORIES', 'BRAND']):
                    continue
                # Extract size pattern (XS, S, M, L, XL, XXL, or numeric sizes)
                size_match = re.search(r'\b(XS|S|M|L|XL|XXL|XXXL|XXS|\d+)\b', size_text, re.IGNORECASE)
                if size_match:
                    return size_match.group(1).upper()
        
        # Look for size in text content near specific keywords
        page_text = soup.get_text()
        # Try to find "Size" followed by actual size values
        size_patterns = [
            r'Size\s*[:]?\s*(\b(?:XS|S|M|L|XL|XXL|XXXL|\d+)\b)',
            r'\b(?:XS|S|M|L|XL|XXL|XXXL|\d+)\b(?=\s*(?:Size|sz))',
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                size_val = match.group(1) if match.lastindex else match.group(0)
                # Validate it's not a false positive
                if size_val and size_val.upper() not in ['BOUTIQUES', 'CATEGORY']:
                    return size_val.strip().upper()
        
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract item description from the page"""
        # Try multiple selectors for description
        selectors = [
            '[data-testid="description"]',
            '.description',
            '[class*="description"]',
            '.listing-description',
            '[itemprop="description"]',
        ]
        
        for selector in selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                description = desc_elem.get_text(strip=True)
                if description and len(description) > 10:  # Ensure meaningful content
                    return description
        
        # Fallback: look for description in meta tags
        meta_desc = soup.find('meta', property='og:description')
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content').strip()
        
        return None
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all image URLs from the listing"""
        images = []
        seen_urls = set()
        
        # Try multiple selectors for listing images
        selectors = [
            'img[data-testid="listing-image"]',
            '[data-testid="listing-image"] img',
            '.listing-image img',
            '.carousel img',
            '.carousel-item img',
            '[class*="carousel"] img',
            '[class*="listing-image"] img',
            '[class*="product-image"] img',
            '[class*="item-image"] img',
            'img[src*="cloudfront"]',  # Poshmark uses CloudFront for images
            'img[src*="/posts/"]',  # Poshmark image path pattern
        ]
        
        for selector in selectors:
            img_elements = soup.select(selector)
            for img in img_elements:
                # Try multiple attributes for image URL
                img_url = (img.get('src') or 
                          img.get('data-src') or 
                          img.get('data-lazy-src') or
                          img.get('data-original') or
                          img.get('data-image-url'))
                
                if img_url:
                    # Convert relative URLs to absolute
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    elif img_url.startswith('/'):
                        img_url = urljoin(base_url, img_url)
                    
                    # Skip if already seen
                    if img_url in seen_urls:
                        continue
                    
                    # Filter criteria
                    skip_patterns = [
                        'logo', 'icon', 'avatar', 'profile', 
                        'button', 'badge', 'emoji', 'svg'
                    ]
                    
                    # Check if URL contains skip patterns (in path, not domain)
                    url_lower = img_url.lower()
                    if any(pattern in url_lower for pattern in skip_patterns):
                        continue
                    
                    # Check if it's likely a product image from the listing
                    # Poshmark listing images are typically on cloudfront with /posts/ path
                    # Filter out collection images, category images, etc.
                    is_product_image = (
                        ('cloudfront' in url_lower and '/posts/' in url_lower) or
                        any(indicator in url_lower for indicator in [
                            'listing', 'product', 'item'
                        ])
                    )
                    
                    # Exclude collection, category, or other non-listing images
                    exclude_patterns = [
                        '/collections/', '/categories/', '/banners/',
                        '/ads/', '/promotions/', 'thumbnail', 'icon'
                    ]
                    if any(pattern in url_lower for pattern in exclude_patterns):
                        continue
                    
                    # Also check image dimensions if available
                    img_width = img.get('width')
                    img_height = img.get('height')
                    # Skip very small images (likely icons)
                    if img_width and img_height:
                        try:
                            width = int(img_width)
                            height = int(img_height)
                            if width < 100 or height < 100:
                                continue
                        except (ValueError, TypeError):
                            pass
                    
                    if is_product_image or 'jpg' in url_lower or 'jpeg' in url_lower or 'png' in url_lower:
                        images.append(img_url)
                        seen_urls.add(img_url)
        
        # Also check for Open Graph images
        og_images = soup.find_all('meta', property='og:image')
        for og_img in og_images:
            img_url = og_img.get('content')
            if img_url and img_url not in seen_urls:
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    img_url = urljoin(base_url, img_url)
                images.append(img_url)
                seen_urls.add(img_url)
        
        # Check for JSON-LD structured data that might contain images
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                # Look for image arrays in JSON-LD
                if isinstance(data, dict):
                    for key in ['image', 'images', 'photo', 'photos']:
                        if key in data:
                            img_data = data[key]
                            if isinstance(img_data, list):
                                for img_item in img_data:
                                    img_url = img_item if isinstance(img_item, str) else img_item.get('url', '')
                                    if img_url and img_url not in seen_urls:
                                        if img_url.startswith('//'):
                                            img_url = 'https:' + img_url
                                        elif img_url.startswith('/'):
                                            img_url = urljoin(base_url, img_url)
                                        images.append(img_url)
                                        seen_urls.add(img_url)
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        
        # Sort images to prioritize main images (often first or largest)
        # Remove duplicates while preserving order
        unique_images = []
        for img_url in images:
            if img_url not in unique_images:
                unique_images.append(img_url)
        
        return unique_images
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract item price from the page"""
        selectors = [
            '[data-testid="price"]',
            '.price',
            '[class*="price"]',
            'span:contains("$")',
        ]
        
        for selector in selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                # Extract price pattern
                price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', price_text)
                if price_match:
                    return price_match.group(0)
        
        return None
    
    def _extract_brand(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract brand name from the page"""
        # Brand is often in the title or as a separate element
        selectors = [
            '[data-testid="brand"]',
            '.brand',
            '[class*="brand"]',
        ]
        
        for selector in selectors:
            brand_elem = soup.select_one(selector)
            if brand_elem:
                brand = brand_elem.get_text(strip=True)
                if brand:
                    return brand
        
        return None
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        """
        Download an image from URL and return as PIL Image
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            PIL Image object or None if download failed
        """
        try:
            self._random_delay()
            headers = self._get_headers()
            headers['Accept'] = 'image/webp,image/apng,image/*,*/*;q=0.8'
            
            response = self.session.get(
                image_url,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                return img
            else:
                return None
                
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def search_poshmark(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search Poshmark for items matching the query and return top results
        
        Args:
            query: Search query string (e.g., "red denim jacket")
            top_k: Number of top results to return (default: 5)
            
        Returns:
            List of dictionaries containing:
                - item_id: Listing ID or URL
                - title: Item title
                - description: Item description (if available from search page)
                - image_url: URL of item image
                - price: Item price (if available)
                - listing_url: Full URL to the listing
                - similarity: Placeholder similarity score (1.0 for now, as these are search results)
        """
        results = []
        
        if not query or not query.strip():
            return results
        
        try:
            # Construct search URL
            # Poshmark search URL format: https://poshmark.com/search?query=...
            encoded_query = quote_plus(query.strip())
            search_url = f"https://poshmark.com/search?query={encoded_query}"
            
            # Add delay before request
            self._random_delay()
            
            # Make request
            headers = self._get_headers()
            response = self.session.get(
                search_url,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code != 200:
                print(f"Search failed with status code: {response.status_code}")
                return results
            
            # Parse HTML
            try:
                html_content = response.text
            except UnicodeDecodeError:
                content_encoding = response.headers.get('Content-Encoding', '').lower()
                if content_encoding == 'br' and BROTLI_AVAILABLE:
                    try:
                        decompressed = brotli.decompress(response.content)
                        html_content = decompressed.decode('utf-8')
                    except Exception as e:
                        print(f"Failed to decompress search results: {e}")
                        return results
                else:
                    html_content = response.content.decode('utf-8', errors='ignore')
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract search results
            # Poshmark search results are typically in cards or tiles
            # Try multiple selectors to find listing cards
            listing_selectors = [
                'a[href*="/listing/"]',  # Direct links - most reliable
                'div[class*="tile"] a[href*="/listing/"]',
                'div[class*="card"] a[href*="/listing/"]',
                '.tile a[href*="/listing/"]',
                '.tile--item a[href*="/listing/"]',
                '[data-testid="tile"] a[href*="/listing/"]',
                '.item-tile a[href*="/listing/"]',
            ]
            
            listing_elements = []
            for selector in listing_selectors:
                elements = soup.select(selector)
                if elements:
                    listing_elements = elements[:top_k * 3]  # Get more than needed
                    break
            
            # Extract listing URLs and basic info from search results
            seen_urls = set()
            listing_data_map = {}  # Map URL to extracted data
            
            # Try to find listing containers - Poshmark uses various structures
            # Look for common container patterns
            container_selectors = [
                'div[class*="tile"]',
                'div[class*="card"]',
                'div[class*="item"]',
                'article',
                '[data-testid*="tile"]',
                '[data-testid*="card"]',
                'a[href*="/listing/"]',  # Direct links to listings
            ]
            
            containers = []
            for selector in container_selectors:
                found = soup.select(selector)
                if found:
                    if selector.startswith('a[href'):
                        # For direct links, use the parent or the link itself
                        containers = found
                    else:
                        # Filter to only those with listing links
                        containers = [c for c in found if c.find('a', href=re.compile(r'/listing/'))]
                    if containers:
                        # Limit to reasonable number but get more than top_k
                        containers = containers[:top_k * 3]
                        break
            
            # If no containers found, use the listing_elements we found earlier
            # Also, if we found direct links, use those as containers
            if not containers and listing_elements:
                containers = listing_elements
            elif not containers:
                # Last resort: find any links to listings
                all_listing_links = soup.select('a[href*="/listing/"]')
                containers = all_listing_links[:top_k * 3] if all_listing_links else []
            
            for container in containers:
                # Find the listing link
                if container.name == 'a':
                    # Container is the link itself
                    link_elem = container
                    href = link_elem.get('href', '')
                else:
                    # Find link within container
                    link_elem = container.find('a', href=re.compile(r'/listing/'))
                    if not link_elem:
                        continue
                    href = link_elem.get('href', '')
                
                if not href:
                    continue
                
                # Normalize URL
                if href.startswith('/'):
                    listing_url = urljoin('https://poshmark.com', href)
                elif href.startswith('http'):
                    listing_url = href
                else:
                    continue
                
                # Skip if we've already processed this URL
                if listing_url in seen_urls:
                    continue
                seen_urls.add(listing_url)
                
                # Stop if we have enough unique URLs
                if len(seen_urls) > top_k * 2:
                    break
                
                # Extract data from the container using multiple strategies
                title = None
                price = None
                image_url = None
                description = None
                
                # Strategy 1: Look for specific data attributes (most reliable)
                title_elem = container.find(attrs={'data-testid': re.compile(r'title|name', re.I)})
                if not title_elem:
                    # Strategy 2: Look for heading tags
                    title_elem = container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if not title_elem:
                    # Strategy 3: Look for aria-label or title attribute
                    title_elem = container.find(attrs={'aria-label': True}) or container.find(attrs={'title': True})
                if title_elem:
                    title = title_elem.get_text(strip=True) or title_elem.get('aria-label') or title_elem.get('title')
                    if title:
                        title = title.strip()
                
                # Extract price with multiple strategies
                price_elem = container.find(attrs={'data-testid': re.compile(r'price', re.I)})
                if not price_elem:
                    price_elem = container.find(attrs={'class': re.compile(r'price', re.I)})
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', price_text)
                    if price_match:
                        price = price_match.group(0)
                
                # If still no price, search in all text
                if not price:
                    container_text = container.get_text()
                    price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', container_text)
                    if price_match:
                        price = price_match.group(0)
                
                # Extract image
                img_elem = container.find('img')
                if img_elem:
                    image_url = (img_elem.get('src') or 
                                img_elem.get('data-src') or 
                                img_elem.get('data-lazy-src') or
                                img_elem.get('data-original'))
                    if image_url:
                        if image_url.startswith('//'):
                            image_url = 'https:' + image_url
                        elif image_url.startswith('/'):
                            image_url = urljoin('https://poshmark.com', image_url)
                
                # Extract description (often limited on search page)
                desc_elem = container.find(attrs={'class': re.compile(r'desc', re.I)})
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
                
                # Store extracted data
                listing_data_map[listing_url] = {
                    'title': title,
                    'price': price,
                    'image_url': image_url,
                    'description': description
                }
            
            # Ensure we have at least top_k listings
            # If we have fewer, try to get more from listing_elements
            if len(listing_data_map) < top_k and listing_elements:
                for elem in listing_elements:
                    if len(listing_data_map) >= top_k:
                        break
                    if elem.name == 'a':
                        href = elem.get('href', '')
                    else:
                        link = elem.find('a', href=True)
                        href = link.get('href', '') if link else ''
                    
                    if href:
                        if href.startswith('/'):
                            listing_url = urljoin('https://poshmark.com', href)
                        elif href.startswith('http'):
                            listing_url = href
                        else:
                            continue
                        
                        if listing_url not in listing_data_map:
                            listing_data_map[listing_url] = {
                                'title': None,
                                'price': None,
                                'image_url': None,
                                'description': None
                            }
            
            # Scrape individual listings to get complete data (especially size)
            # Process all listings up to top_k
            processed_count = 0
            for listing_url, extracted_data in list(listing_data_map.items())[:top_k]:
                if processed_count >= top_k:
                    break
                
                # Always scrape to get size and other missing fields
                # We need size, so always do at least a partial scrape
                needs_full_scrape = (
                    not extracted_data.get('title') or 
                    not extracted_data.get('price') or 
                    not extracted_data.get('size')
                )
                
                # Always scrape to ensure we get size and complete data
                try:
                    listing_data = self.scrape_listing(listing_url)
                    if listing_data.get('success'):
                        # Merge scraped data with extracted data (prefer scraped)
                        extracted_data['title'] = listing_data.get('title') or extracted_data.get('title')
                        extracted_data['price'] = listing_data.get('price') or extracted_data.get('price')
                        extracted_data['description'] = listing_data.get('description') or extracted_data.get('description')
                        if not extracted_data.get('image_url') and listing_data.get('images'):
                            extracted_data['image_url'] = listing_data.get('images', [None])[0]
                        # Always get size from full scrape
                        extracted_data['size'] = listing_data.get('size') or extracted_data.get('size')
                        extracted_data['brand'] = listing_data.get('brand') or extracted_data.get('brand')
                except Exception as e:
                    print(f"Error scraping listing {listing_url}: {e}")
                    # Continue with whatever data we have - still add the result
                
                # Create result item
                item_id = listing_url.split('/')[-1] if listing_url else f"item_{len(results)}"
                result_item = {
                    "item_id": item_id,
                    "title": extracted_data.get('title') or "Untitled Item",
                    "description": extracted_data.get('description') or extracted_data.get('title') or "No description available",
                    "image_url": extracted_data.get('image_url'),
                    "price": extracted_data.get('price'),
                    "size": extracted_data.get('size'),
                    "brand": extracted_data.get('brand'),
                    "listing_url": listing_url,
                    "similarity": 1.0
                }
                results.append(result_item)
                processed_count += 1
            
            return results[:top_k]
            
        except requests.exceptions.Timeout:
            print("Search request timed out")
            return results
        except requests.exceptions.RequestException as e:
            print(f"Network error during search: {e}")
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            return results

