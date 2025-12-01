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
        self.timeout = 15   # Request timeout (seconds)
        
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
            try:
                html_content = response.text
            except UnicodeDecodeError:
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
            
            # Extract data
            title = self._extract_title(soup)
            result['title'] = title
            
            size = self._extract_size(soup)
            result['size'] = size
            
            description = self._extract_description(soup)
            result['description'] = description
            
            images = self._extract_images(soup, url)
            result['images'] = images
            
            price = self._extract_price(soup)
            result['price'] = price
            
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
        selectors = [
            'h1[data-testid="listing-title"]', 'h1.listing-title', 'h1.title', 'h1',
            '[data-testid="title"]', '.title h1',
        ]
        for selector in selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title: return title
        
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title.get('content').strip()
        return None
    
    def _extract_size(self, soup: BeautifulSoup) -> Optional[str]:
        selectors = [
            '[data-testid="size"]', '[class*="size"][class*="value"]',
            'span:contains("Size"):not(:contains("BOUTIQUES"))',
        ]
        for selector in selectors:
            size_elem = soup.select_one(selector)
            if size_elem:
                size_text = size_elem.get_text(strip=True)
                if any(skip in size_text.upper() for skip in ['BOUTIQUES', 'CATEGORY', 'CATEGORIES', 'BRAND']):
                    continue
                size_match = re.search(r'\b(XS|S|M|L|XL|XXL|XXXL|XXS|\d+)\b', size_text, re.IGNORECASE)
                if size_match:
                    return size_match.group(1).upper()
        
        page_text = soup.get_text()
        size_patterns = [
            r'Size\s*[:]?\s*(\b(?:XS|S|M|L|XL|XXL|XXXL|\d+)\b)',
            r'\b(?:XS|S|M|L|XL|XXL|XXXL|\d+)\b(?=\s*(?:Size|sz))',
        ]
        for pattern in size_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                size_val = match.group(1) if match.lastindex else match.group(0)
                if size_val and size_val.upper() not in ['BOUTIQUES', 'CATEGORY']:
                    return size_val.strip().upper()
        return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        selectors = [
            '[data-testid="description"]', '.description', '[class*="description"]',
            '.listing-description', '[itemprop="description"]',
        ]
        for selector in selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                description = desc_elem.get_text(strip=True)
                if description and len(description) > 10:
                    return description
        
        meta_desc = soup.find('meta', property='og:description')
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content').strip()
        return None
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        images = []
        seen_urls = set()
        
        selectors = [
            'img[data-testid="listing-image"]', '[data-testid="listing-image"] img',
            '.listing-image img', '.carousel img', '.carousel-item img',
            '[class*="carousel"] img', '[class*="listing-image"] img',
            '[class*="product-image"] img', '[class*="item-image"] img',
            'img[src*="cloudfront"]', 'img[src*="/posts/"]',
        ]
        
        for selector in selectors:
            img_elements = soup.select(selector)
            for img in img_elements:
                img_url = (img.get('src') or img.get('data-src') or 
                          img.get('data-lazy-src') or img.get('data-original') or
                          img.get('data-image-url'))
                
                if img_url:
                    if img_url.startswith('//'): img_url = 'https:' + img_url
                    elif img_url.startswith('/'): img_url = urljoin(base_url, img_url)
                    
                    if img_url in seen_urls: continue
                    
                    skip_patterns = ['logo', 'icon', 'avatar', 'profile', 'button', 'badge', 'emoji', 'svg']
                    url_lower = img_url.lower()
                    if any(pattern in url_lower for pattern in skip_patterns): continue
                    
                    is_product_image = (
                        ('cloudfront' in url_lower and '/posts/' in url_lower) or
                        any(indicator in url_lower for indicator in ['listing', 'product', 'item'])
                    )
                    
                    exclude_patterns = ['/collections/', '/categories/', '/banners/', '/ads/', '/promotions/', 'thumbnail', 'icon']
                    if any(pattern in url_lower for pattern in exclude_patterns): continue
                    
                    if is_product_image or 'jpg' in url_lower or 'jpeg' in url_lower or 'png' in url_lower:
                        images.append(img_url)
                        seen_urls.add(img_url)
        
        # Sort images to prioritize main images (remove duplicates while preserving order)
        unique_images = []
        for img_url in images:
            if img_url not in unique_images:
                unique_images.append(img_url)
        
        return unique_images
    
    def _extract_price(self, soup: BeautifulSoup) -> Optional[str]:
        selectors = ['[data-testid="price"]', '.price', '[class*="price"]', 'span:contains("$")']
        for selector in selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', price_text)
                if price_match: return price_match.group(0)
        return None
    
    def _extract_brand(self, soup: BeautifulSoup) -> Optional[str]:
        selectors = ['[data-testid="brand"]', '.brand', '[class*="brand"]']
        for selector in selectors:
            brand_elem = soup.select_one(selector)
            if brand_elem:
                brand = brand_elem.get_text(strip=True)
                if brand: return brand
        return None
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        try:
            self._random_delay()
            headers = self._get_headers()
            headers['Accept'] = 'image/webp,image/apng,image/*,*/*;q=0.8'
            
            response = self.session.get(
                image_url, headers=headers, timeout=self.timeout, stream=True
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
        Search Poshmark for items matching the query and return top results.
        Uses exact query without stripping punctuation.
        """
        results = []
        
        if not query or not query.strip():
            return results
        
        try:
            # Construct search URL - Using the exact query string
            # We strip whitespace but do NOT remove other characters
            encoded_query = quote_plus(query.strip())
            
            # Use sort_by=relevance to get best matches
            search_url = f"https://poshmark.com/search?query={encoded_query}&sort_by=relevance"
            
            print(f"DEBUG: Searching Poshmark URL: {search_url}")
            
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
            
            # --- IMPROVED FINDING LOGIC ---
            # 1. Select all <a> tags that look like listing links
            listing_links = soup.select('a[href*="/listing/"]')
            
            # 2. Deduplicate links
            seen_urls = set()
            unique_links = []
            
            for link in listing_links:
                href = link.get('href', '')
                if href in seen_urls: continue
                
                # Filter out irrelevant user/party/brand links
                if '/user/' in href or '/party/' in href or '/brand/' in href:
                    continue
                
                seen_urls.add(href)
                unique_links.append(link)
            
            # 3. Process exactly top_k items
            # We grab a few extras just in case some fail processing
            for link_elem in unique_links[:top_k * 2]:
                if len(results) >= top_k:
                    break

                # Get URL
                href = link_elem.get('href', '')
                if href.startswith('/'): listing_url = urljoin('https://poshmark.com', href)
                elif href.startswith('http'): listing_url = href
                else: continue
                
                # Determine "Card" Context to find metadata
                # We search up the HTML tree to find the box wrapping this link
                card = link_elem.find_parent(attrs={'class': re.compile(r'card|tile|item|article', re.I)})
                context = card if card else link_elem

                # Extract Data from the Card/Context
                title = None
                price = None
                image_url = None
                
                # Title
                title = link_elem.get('title')
                if not title:
                    title_elem = context.find(attrs={'data-testid': 'listing-title'}) or \
                                 context.find(attrs={'class': re.compile(r'title', re.I)})
                    if title_elem: title = title_elem.get_text(strip=True)
                
                # Price
                price_elem = context.find(attrs={'data-testid': 'listing-price'}) or \
                             context.find(attrs={'class': re.compile(r'price', re.I)}) or \
                             context.find(string=re.compile(r'\$[\d,]+'))
                
                if price_elem:
                    p_text = price_elem if isinstance(price_elem, str) else price_elem.get_text(strip=True)
                    p_match = re.search(r'\$[\d,]+', p_text)
                    if p_match: price = p_match.group(0)

                # Image
                img_elem = context.find('img')
                if img_elem:
                    image_url = (img_elem.get('src') or img_elem.get('data-src') or 
                               img_elem.get('data-lazy-src'))
                    
                if not title: title = "Untitled Item"
                
                results.append({
                    "item_id": listing_url.split('/')[-1].split('?')[0],
                    "title": title,
                    "description": title, 
                    "image_url": image_url,
                    "price": price,
                    "listing_url": listing_url,
                    "brand": "Poshmark Find",
                    "similarity": "Search Match" 
                })
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return results