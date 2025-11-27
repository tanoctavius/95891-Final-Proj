"""
Poshmark Listing Scraper using Selenium for JavaScript-rendered content
This version uses a headless browser to properly render Poshmark pages
"""

import time
import random
from typing import Dict, Optional, List
from PIL import Image
from io import BytesIO
import re
from urllib.parse import urlparse

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Install with: pip install selenium")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class PoshmarkScraperSelenium:
    """Scraper for Poshmark listings using Selenium for JavaScript rendering"""
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self, headless: bool = True):
        """Initialize scraper with Selenium WebDriver"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required. Install with: pip install selenium")
        
        self.headless = headless
        self.driver = None
        self.min_delay = 2
        self.max_delay = 5
        self._init_driver()
    
    def _init_driver(self):
        """Initialize Chrome WebDriver with options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # Use random user agent
        user_agent = random.choice(self.USER_AGENTS)
        chrome_options.add_argument(f'user-agent={user_agent}')
        
        try:
            # Try to use ChromeDriver (must be installed separately)
            self.driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromeDriver: {e}\n"
                "Please install ChromeDriver: https://chromedriver.chromium.org/"
            )
    
    def _random_delay(self):
        """Add random delay between actions"""
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
        Scrape a Poshmark listing using Selenium
        
        Args:
            url: Poshmark listing URL
            
        Returns:
            Dictionary with scraped data
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
        
        if not self._validate_url(url):
            result['error'] = 'Invalid Poshmark URL'
            return result
        
        try:
            # Navigate to page
            self.driver.get(url)
            
            # Wait for page to load (wait for title element or specific content)
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                result['error'] = 'Page took too long to load'
                return result
            
            # Wait a bit more for JavaScript to render content
            self._random_delay()
            
            # Get page source after JavaScript rendering
            page_source = self.driver.page_source
            
            if not BS4_AVAILABLE:
                # Try to extract using Selenium selectors only
                result = self._extract_with_selenium(result)
            else:
                # Use BeautifulSoup on the rendered HTML
                soup = BeautifulSoup(page_source, 'html.parser')
                result = self._extract_with_bs4(soup, result, url)
            
            # Mark as successful if we got at least title or description
            if result['title'] or result['description']:
                result['success'] = True
            else:
                result['error'] = 'Could not extract item information from page'
                
        except WebDriverException as e:
            result['error'] = f'WebDriver error: {str(e)}'
        except Exception as e:
            result['error'] = f'Unexpected error: {str(e)}'
        
        return result
    
    def _extract_with_selenium(self, result: Dict) -> Dict:
        """Extract data using Selenium selectors"""
        try:
            # Try to find title
            title_selectors = [
                "h1[data-testid='listing-title']",
                "h1.listing-title",
                "h1.title",
                "h1"
            ]
            for selector in title_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element:
                        result['title'] = element.text.strip()
                        break
                except:
                    continue
        except:
            pass
        
        return result
    
    def _extract_with_bs4(self, soup, result: Dict, url: str) -> Dict:
        """Extract data using BeautifulSoup"""
        # Extract title
        title_selectors = [
            'h1[data-testid="listing-title"]',
            'h1.listing-title',
            'h1.title',
            'h1',
            '[data-testid="title"]',
        ]
        
        for selector in title_selectors:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                result['title'] = elem.get_text(strip=True)
                break
        
        # Extract size
        size_selectors = [
            '[data-testid="size"]',
            '.size',
            '[class*="size"]',
        ]
        
        for selector in size_selectors:
            elem = soup.select_one(selector)
            if elem:
                size_text = elem.get_text(strip=True)
                size_match = re.search(r'\b(XS|S|M|L|XL|XXL|XXXL|\d+)\b', size_text, re.IGNORECASE)
                if size_match:
                    result['size'] = size_match.group(1).upper()
                    break
        
        # Extract description
        desc_selectors = [
            '[data-testid="description"]',
            '.description',
            '[class*="description"]',
            '.listing-description',
        ]
        
        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem:
                desc = elem.get_text(strip=True)
                if desc and len(desc) > 10:
                    result['description'] = desc
                    break
        
        # Extract images
        img_elements = soup.select('img[src*="poshmark"]')
        seen_urls = set()
        
        for img in img_elements:
            img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if img_url and img_url not in seen_urls:
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif img_url.startswith('/'):
                    from urllib.parse import urljoin
                    img_url = urljoin(url, img_url)
                
                if 'logo' not in img_url.lower():
                    result['images'].append(img_url)
                    seen_urls.add(img_url)
        
        # Extract price
        price_elements = soup.select('[data-testid="price"], .price')
        for elem in price_elements:
            price_text = elem.get_text(strip=True)
            price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', price_text)
            if price_match:
                result['price'] = price_match.group(0)
                break
        
        return result
    
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()

