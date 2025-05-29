import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
import time
import hashlib
from datetime import datetime
from typing import Set, List, Optional, Dict, Any
import re
from database import CrawlerDatabase
from playwright.sync_api import sync_playwright, Browser, Page

class WebCrawler:
    def __init__(self, db_path: str = "crawler.db", delay: float = 1.0, 
                 user_agent: str = "WebCrawler/1.0", use_javascript: bool = False):
        self.db = CrawlerDatabase(db_path)
        self.delay = delay
        self.user_agent = user_agent
        self.use_javascript = use_javascript
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.robots_cache = {}
        self.running = False
        self.debug = False
        
        # Setup Playwright if needed
        if use_javascript:
            self.setup_playwright()
    
    def setup_playwright(self):
        """Setup Playwright for JavaScript support"""
        try:
            self.playwright = sync_playwright().start()
            
            # Launch browser with more realistic settings
            self.browser = self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-features=TranslateUI',
                    '--disable-ipc-flooding-protection',
                    '--disable-blink-features=AutomationControlled',  # Hide automation
                    '--no-first-run',
                    '--no-default-browser-check'
                ]
            )
            
            # Create a context with more realistic browser settings
            self.context = self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                # Add some browser features that sites might check for
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                }
            )
            
            # Add JavaScript to hide automation indicators
            self.context.add_init_script("""
                // Hide webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Mock plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
                
                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            print("Playwright browser initialized successfully with enhanced settings")
            
        except Exception as e:
            print(f"Failed to initialize Playwright: {e}")
            self.use_javascript = False
            self.playwright = None
            self.browser = None
            self.context = None
    
    def _get_robots_parser(self, base_url: str) -> RobotFileParser:
        """Get robots.txt parser for domain"""
        parsed_url = urlparse(base_url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        if domain not in self.robots_cache:
            robots_url = urljoin(domain, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
                self.robots_cache[domain] = rp
            except:
                # If robots.txt can't be fetched, create permissive parser
                rp = RobotFileParser()
                self.robots_cache[domain] = rp
        
        return self.robots_cache[domain]
    
    def _can_fetch(self, url: str, base_url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        try:
            rp = self._get_robots_parser(base_url)
            return rp.can_fetch(self.user_agent, url)
        except:
            return True  # If we can't check, assume it's okay
    
    def _is_same_domain(self, url: str, base_url: str) -> bool:
        """Check if URL is from the same domain as base URL"""
        base_domain = urlparse(base_url).netloc
        url_domain = urlparse(url).netloc
        return url_domain == base_domain or url_domain.endswith('.' + base_domain)
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments, query parameters, and standardizing case/slashes"""
        parsed = urlparse(url)
        
        # Convert path to lowercase and remove trailing slash
        path = parsed.path.lower()
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        
        # Remove fragment and query parameters, normalize case
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(), 
            path,
            parsed.params, 
            '', ''  # Remove query and fragment
        ))
        return normalized
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Remove specific div IDs that should be excluded
        specific_ids_to_remove = ['top-navigation', 'header-container', 'footer-container']
        for div_id in specific_ids_to_remove:
            div_element = soup.find(id=div_id)
            if div_element:
                if self.debug:
                    print(f"  -> Removing div with ID: {div_id}")
                div_element.decompose()
        
        # Remove specific div classes that should be excluded
        specific_classes_to_remove = ['left-side cf']
        for div_class in specific_classes_to_remove:
            div_elements = soup.find_all('div', class_=div_class)
            for div_element in div_elements:
                if self.debug:
                    print(f"  -> Removing div with class: {div_class}")
                div_element.decompose()
        
        # Remove elements that commonly contain JavaScript disabled messages
        js_disabled_selectors = [
            'noscript',
            '[class*="javascript"]',
            '[class*="js-disabled"]',
            '[class*="no-js"]',
            '[id*="javascript"]',
            '[id*="js-disabled"]',
            '[id*="no-js"]',
            '.js-disabled',
            '#js-disabled'
        ]
        
        for selector in js_disabled_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove any divs or spans that contain JavaScript disabled messages
        elements_to_remove = []
        for element in soup.find_all(['div', 'span', 'p', 'section']):
            if element.get_text():
                text = element.get_text().strip()
                if self._has_js_disabled_message(text) and len(text) < 500:  # Only remove short JS disabled messages
                    elements_to_remove.append(element)
                    if self.debug:
                        print(f"  -> Removing JS disabled message element: {text[:100]}...")
        
        for element in elements_to_remove:
            element.decompose()
        
        # Try to find main content areas first - be more specific for JEA site
        main_content = None
        
        # Look for common content containers
        content_selectors = [
            'main',
            'article', 
            '[role="main"]',
            '.main-content',
            '.content',
            '.page-content',
            '#main-content',
            '#content',
            '.container .row',  # Bootstrap-style layouts
            '.content-wrapper'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                if self.debug:
                    print(f"  -> Found main content using selector: {selector}")
                break
        
        # If no main content found, try to exclude header/footer/nav
        if not main_content:
            # Remove navigation, header, footer elements
            for element in soup(['nav', 'header', 'footer']):
                element.decompose()
            
            # Remove elements with navigation-related classes
            nav_selectors = [
                '[class*="nav"]',
                '[class*="menu"]', 
                '[class*="header"]',
                '[class*="footer"]',
                '[class*="sidebar"]'
            ]
            
            for selector in nav_selectors:
                for element in soup.select(selector):
                    # Only remove if it's likely navigation (short text, many links)
                    text = element.get_text().strip()
                    links = element.find_all('a')
                    if len(text) < 200 and len(links) > 3:
                        element.decompose()
            
            main_content = soup
        
        # NOW convert hyperlinks to markdown format (after link extraction is done)
        for link in main_content.find_all('a', href=True):
            href = link['href']
            link_text = link.get_text().strip()
            
            # Skip empty links or links without text
            if not link_text:
                continue
            
            # Skip certain types of links
            if href.startswith(('mailto:', 'tel:', 'javascript:')):
                continue
            
            # Convert relative URLs to absolute URLs
            if hasattr(self, '_current_url'):
                absolute_href = urljoin(self._current_url, href)
            else:
                absolute_href = href
            
            # Create markdown link
            markdown_link = f"[{link_text}]({absolute_href})"
            
            # Replace the link element with the markdown text
            link.replace_with(markdown_link)
        
        # Extract text from the main content
        if main_content:
            text = main_content.get_text()
        else:
            text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Final filter: remove any remaining JavaScript disabled messages
        sentences = text.split('.')
        filtered_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not self._has_js_disabled_message(sentence):
                filtered_sentences.append(sentence)
        
        text = '. '.join(filtered_sentences)
        
        # Additional cleanup for common non-content text
        cleanup_patterns = [
            r'Skip to main content',
            r'Skip navigation',
            r'JavaScript must be enabled',
            r'This site requires JavaScript',
            r'Please enable JavaScript',
            r'JavaScript is disabled',
            r'Enable JavaScript to view this site',
            r'Your browser does not support JavaScript'
        ]
        
        for pattern in cleanup_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If the text is very short, it might be mostly JS disabled messages
        if len(text) < 100 and self.debug:
            print(f"  -> Warning: Extracted text is very short ({len(text)} chars)")
            print(f"  -> Text preview: {text[:200]}...")
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, current_url: str) -> List[str]:
        """Extract internal links from HTML"""
        links = []
        all_links_found = []
        
        # Define URL patterns to exclude
        excluded_patterns = [
            '/events/',
            '/event/',
        ]
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            all_links_found.append(href)
            
            # Skip mailto, tel, javascript links
            if href.startswith(('mailto:', 'tel:', 'javascript:')):
                if self.debug:
                    print(f"    -> Skipping special link: {href}")
                continue
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(current_url, href)
            normalized_url = self._normalize_url(absolute_url)
            
            # Check for excluded URL patterns
            excluded = False
            for pattern in excluded_patterns:
                if pattern in normalized_url:
                    excluded = True
                    if self.debug:
                        print(f"    -> Skipping excluded URL pattern '{pattern}': {normalized_url}")
                    break
            
            if excluded:
                continue
            
            # Only include links from the same domain
            if self._is_same_domain(normalized_url, base_url):
                links.append(normalized_url)
                if self.debug:
                    print(f"    -> Added internal link: {normalized_url}")
            else:
                if self.debug:
                    print(f"    -> Skipping external link: {normalized_url}")
        
        # Debug output
        if self.debug:
            print(f"  -> Found {len(all_links_found)} total links, {len(links)} internal links")
            if len(all_links_found) > 0:
                print(f"  -> Sample links found: {all_links_found[:5]}")
        else:
            print(f"  -> Found {len(all_links_found)} total links, {len(links)} internal links")
        
        return list(set(links))  # Remove duplicates
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate MD5 hash of content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _fetch_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch URL with optional JavaScript support"""
        if self.use_javascript and hasattr(self, 'context') and self.context:
            return self._fetch_url_playwright(url)
        else:
            return self._fetch_url_requests(url)
    
    def _fetch_url_requests(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch URL using requests (no JavaScript)"""
        try:
            # Store current URL for link resolution
            self._current_url = url
            
            response = self.session.get(url, timeout=30)
            
            # Get last modified date if available
            last_modified = None
            if 'Last-Modified' in response.headers:
                try:
                    last_modified = datetime.strptime(
                        response.headers['Last-Modified'], 
                        '%a, %d %b %Y %H:%M:%S %Z'
                    )
                except:
                    pass
            
            return {
                'status_code': response.status_code,
                'content': response.text if response.status_code == 200 else '',
                'headers': dict(response.headers),
                'last_modified': last_modified
            }
        
        except Exception as e:
            return {
                'status_code': 0,
                'content': '',
                'headers': {},
                'last_modified': None,
                'error': str(e)
            }
    
    def _fetch_url_playwright(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch URL using Playwright (with JavaScript support)"""
        page = None
        try:
            # Store current URL for link resolution
            self._current_url = url
            
            # Create new page
            page = self.context.new_page()
            
            if self.debug:
                # Enable console logging in debug mode
                page.on("console", lambda msg: print(f"  -> Browser console: {msg.text}"))
            
            # Set longer timeout for slow pages
            page.set_default_timeout(60000)  # 60 seconds
            
            # Block unnecessary resources to speed up loading (but allow more JS)
            page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", lambda route: route.abort())
            page.route("**/ads/**", lambda route: route.abort())
            page.route("**/analytics/**", lambda route: route.abort())
            page.route("**/tracking/**", lambda route: route.abort())
            
            # Navigate to URL and wait for network to be idle
            response = page.goto(url, wait_until='networkidle', timeout=60000)
            
            # Wait for initial JavaScript to execute
            page.wait_for_timeout(3000)
            
            # Try to wait for common indicators that the page is fully loaded
            try:
                # Wait for jQuery if it's being used
                page.wait_for_function("() => typeof window.$ !== 'undefined' || typeof window.jQuery !== 'undefined'", timeout=10000)
                if self.debug:
                    print("  -> jQuery detected and loaded")
            except:
                if self.debug:
                    print("  -> jQuery not detected or timeout")
            
            # Wait for document ready state
            try:
                page.wait_for_function("() => document.readyState === 'complete'", timeout=15000)
                if self.debug:
                    print("  -> Document ready state is complete")
            except:
                if self.debug:
                    print("  -> Document ready state timeout")
            
            # Additional wait for dynamic content
            page.wait_for_timeout(5000)
            
            # Get initial content
            content = page.content()
            
            # If we detect JavaScript disabled message, try more aggressive approaches
            if self._has_js_disabled_message(content):
                if self.debug:
                    print(f"  -> JavaScript disabled message detected, trying enhanced approach...")
                
                # Try to trigger any lazy loading or dynamic content
                try:
                    # Scroll to bottom and back to top to trigger lazy loading
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(2000)
                    
                    # Try to trigger common events that might load content
                    page.evaluate("""
                        // Trigger various events that might load content
                        window.dispatchEvent(new Event('load'));
                        window.dispatchEvent(new Event('DOMContentLoaded'));
                        window.dispatchEvent(new Event('resize'));
                        
                        // Try to trigger any click events on the body
                        document.body.click();
                        
                        // If there are any buttons or links that might enable content, try clicking them
                        const enableButtons = document.querySelectorAll('button, a, div[role="button"]');
                        enableButtons.forEach((btn, index) => {
                            if (index < 5) { // Only try first 5 to avoid too many clicks
                                const text = btn.textContent.toLowerCase();
                                if (text.includes('continue') || text.includes('proceed') || 
                                    text.includes('enable') || text.includes('allow') ||
                                    text.includes('accept') || text.includes('ok')) {
                                    try {
                                        btn.click();
                                    } catch(e) {}
                                }
                            }
                        });
                        
                        // Try to remove any overlay divs that might be blocking content
                        const overlays = document.querySelectorAll('div[style*="position: fixed"], div[style*="position: absolute"]');
                        overlays.forEach(overlay => {
                            const style = window.getComputedStyle(overlay);
                            if (style.zIndex > 1000) {
                                overlay.style.display = 'none';
                            }
                        });
                    """)
                    
                    page.wait_for_timeout(3000)
                    
                    # Try to wait for any new content to load
                    try:
                        page.wait_for_function(
                            "() => document.body.innerText.length > 1000", 
                            timeout=10000
                        )
                        if self.debug:
                            print("  -> Content appears to have loaded (text length > 1000)")
                    except:
                        if self.debug:
                            print("  -> Content length check timeout")
                    
                    # Get content again
                    content = page.content()
                    
                except Exception as e:
                    if self.debug:
                        print(f"  -> Error in enhanced JavaScript execution: {e}")
            
            # Final attempt: if still getting JS disabled, try to find and execute any inline scripts
            if self._has_js_disabled_message(content):
                if self.debug:
                    print(f"  -> Still getting JS disabled message, trying to execute inline scripts...")
                
                try:
                    # Find and execute any inline scripts that might not have run
                    page.evaluate("""
                        // Re-execute any inline scripts
                        const scripts = document.querySelectorAll('script:not([src])');
                        scripts.forEach(script => {
                            if (script.textContent && script.textContent.trim()) {
                                try {
                                    eval(script.textContent);
                                } catch(e) {
                                    console.log('Script execution error:', e);
                                }
                            }
                        });
                        
                        // Try to manually trigger any onload events
                        if (window.onload) {
                            window.onload();
                        }
                        
                        // Trigger jQuery ready if available
                        if (window.$ && window.$.ready) {
                            window.$(document).ready();
                        }
                    """)
                    
                    page.wait_for_timeout(5000)
                    content = page.content()
                    
                except Exception as e:
                    if self.debug:
                        print(f"  -> Error executing inline scripts: {e}")
            
            # Get final page state
            title = page.title()
            final_url = page.url
            
            # Get response status
            status_code = response.status if response else 200
            
            # Try to get headers from the response
            headers = {}
            if response:
                headers = response.headers
            
            # Debug: Save content if still problematic
            if self.debug and self._has_js_disabled_message(content):
                timestamp = int(time.time())
                with open(f"debug_content_{timestamp}.html", "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"  -> Still has JS disabled message, saved to debug_content_{timestamp}.html")
            
            return {
                'status_code': status_code,
                'content': content,
                'headers': headers,
                'last_modified': None,
                'final_url': final_url,
                'title': title
            }
            
        except Exception as e:
            print(f"  -> Playwright error: {e}")
            return {
                'status_code': 0,
                'content': '',
                'headers': {},
                'last_modified': None,
                'error': str(e)
            }
        
        finally:
            # Always close the page to prevent memory leaks
            if page:
                try:
                    page.close()
                except:
                    pass
    
    def _has_js_disabled_message(self, content: str) -> bool:
        """Check if content contains JavaScript disabled messages"""
        if not content:
            return False
        
        js_disabled_indicators = [
            "javascript disabled",
            "javascript is disabled", 
            "enable javascript",
            "javascript must be enabled",
            "this site will not function properly without javascript",
            "please enable javascript",
            "javascript is required",
            "turn on javascript",
            "javascript needs to be enabled",
            "requires javascript",
            "without javascript enabled",
            "browser settings",
            "adjust your browser settings"
        ]
        
        content_lower = content.lower()
        
        # Check for exact phrases
        for indicator in js_disabled_indicators:
            if indicator in content_lower:
                return True
        
        # Check for patterns like "JavaScript" + "disabled/enabled/required"
        if 'javascript' in content_lower:
            js_related_words = ['disabled', 'enabled', 'required', 'must', 'please', 'browser', 'settings']
            for word in js_related_words:
                if word in content_lower:
                    return True
        
        return False
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if extracted text has meaningful content beyond JS disabled messages"""
        if len(text) < 50:
            return False
        
        # Check if it's mostly JavaScript disabled messages
        words = text.lower().split()
        if len(words) == 0:
            return False
        
        js_disabled_ratio = 0
        js_keywords = ['javascript', 'disabled', 'enabled', 'browser', 'settings', 'enable', 'required', 'function', 'properly']
        
        for word in words:
            if any(keyword in word for keyword in js_keywords):
                js_disabled_ratio += 1
        
        # If more than 20% of words are JS-related, consider it not meaningful
        if (js_disabled_ratio / len(words)) > 0.2:
            if self.debug:
                print(f"  -> Content appears to be mostly JS disabled messages ({js_disabled_ratio}/{len(words)} JS-related words)")
            return False
        
        # Check for actual content indicators
        content_indicators = ['about', 'services', 'contact', 'information', 'company', 'business', 'customer', 'electric', 'utility', 'power', 'energy']
        content_score = 0
        
        for word in words:
            if any(indicator in word.lower() for indicator in content_indicators):
                content_score += 1
        
        # If we have some content indicators, it's probably meaningful
        if content_score > 2:
            return True
        
        # If text is long enough and not mostly JS messages, consider it meaningful
        return len(text) > 200
    
    def start_crawl(self, base_url: str) -> int:
        """Start crawling a website"""
        # Add source to database
        source_id = self.db.add_source('website', base_url=base_url)
        
        # Add base URL to queue
        self.db.add_to_queue(source_id, base_url, priority=10)
        
        # Log start
        self.db.log_event('crawl_started', f'Started crawling {base_url}', source_id=source_id)
        
        return source_id
    
    def crawl_source(self, source_id: int, max_pages: Optional[int] = None) -> bool:
        """Crawl all URLs for a source"""
        try:
            self.running = True
            pages_crawled = 0
            
            # Update source status
            start_time = datetime.now()
            self.db.update_source_status(source_id, 'running', start_time=start_time)
            
            self.db.log_event('crawl_started', f'Starting crawl (max_pages: {max_pages})', 
                             source_id=source_id)
            
            while self.running:
                # Get next URL from queue
                queue_item = self.db.get_next_queue_item(source_id)
                if not queue_item:
                    print(f"No more URLs in queue. Total pages crawled: {pages_crawled}")
                    break
                
                queue_id, url = queue_item
                
                # Check if we've hit the page limit
                if max_pages and pages_crawled >= max_pages:
                    print(f"Reached max pages limit ({max_pages})")
                    break
                
                print(f"Fetching: {url}")
                
                # Mark as processing
                self.db.mark_queue_item_processing(queue_id)
                
                # Fetch the URL
                response_data = self._fetch_url(url)
                
                if response_data and response_data['status_code'] == 200:
                    # Parse HTML
                    soup = BeautifulSoup(response_data['content'], 'html.parser')
                    
                    # Extract links FIRST (before modifying the soup for content extraction)
                    base_url = self.db.get_source_base_url(source_id)
                    links = self._extract_links(soup, base_url, response_data.get('final_url', url))
                    
                    # Then extract content (which will modify the soup by converting links to markdown)
                    title = response_data.get('title') or (soup.title.string.strip() if soup.title else None)
                    text_content = self._extract_text_content(soup)
                    content_hash = self._calculate_content_hash(response_data['content'])
                    
                    # Check if we got meaningful content
                    is_meaningful = self._has_meaningful_content(text_content)
                    if not is_meaningful:
                        if self.debug:
                            print(f"  -> Warning: Content appears to be mostly JavaScript disabled messages")
                            print(f"  -> Text preview: {text_content[:200]}...")
                    else:
                        if self.debug:
                            print(f"  -> Extracted {len(text_content)} characters of meaningful content")
                    
                    # Save document
                    doc_id = self.db.upsert_document(
                        source_id=source_id,
                        url=response_data.get('final_url', url),
                        content_hash=content_hash,
                        extracted_text=text_content,
                        title=title,
                        http_status_code=response_data['status_code'],
                        last_modified=response_data['last_modified'],
                        metadata={
                            'headers': response_data['headers'],
                            'meaningful_content': is_meaningful,
                            'content_length': len(text_content)
                        }
                    )
                    
                    # Add links to queue
                    links_added = 0
                    for link in links:
                        if self.db.add_to_queue(source_id, link):
                            links_added += 1
                    
                    print(f"  -> Added {links_added} new URLs to queue")
                    
                    self.db.log_event('url_processed', 
                                    f'Successfully processed: {url} (found {len(links)} links, meaningful: {is_meaningful})',
                                    source_id=source_id, document_id=doc_id)
                    
                    pages_crawled += 1
                    print(f"  -> Processed successfully ({pages_crawled} pages so far, meaningful: {is_meaningful})")
                
                else:
                    # Handle error
                    self.db.upsert_document(
                        source_id=source_id,
                        url=url,
                        content_hash='',
                        extracted_text='',
                        http_status_code=response_data['status_code'] if response_data else 0
                    )
                    
                    error_msg = response_data.get('error', f'HTTP {response_data["status_code"]}') if response_data else 'Unknown error'
                    self.db.log_event('url_error', f'Error fetching {url}: {error_msg}',
                                    source_id=source_id)
                    print(f"  -> Error: {error_msg}")
                
                # Mark queue item as completed
                self.db.mark_queue_item_completed(queue_id)
                
                # Respect crawl delay
                time.sleep(self.delay)
            
            # Update source status
            finish_time = datetime.now()
            status = 'completed' if not self.running else 'stopped'
            self.db.update_source_status(source_id, status, finish_time=finish_time)
            
            self.db.log_event('crawl_finished', 
                            f'Crawl finished. Pages processed: {pages_crawled}',
                            source_id=source_id)
            
            return True
            
        except Exception as e:
            self.db.update_source_status(source_id, 'error')
            self.db.log_event('crawl_error', f'Crawl error: {str(e)}', source_id=source_id)
            print(f"Crawl error: {e}")
            return False
    
    def stop_crawl(self):
        """Stop the current crawl"""
        self.running = False
    
    def get_stats(self, source_id: int) -> Dict[str, Any]:
        """Get crawling statistics"""
        return self.db.get_crawl_stats(source_id)
    
    def close(self):
        """Clean up Playwright resources"""
        try:
            if hasattr(self, 'context') and self.context:
                self.context.close()
            if hasattr(self, 'browser') and self.browser:
                self.browser.close()
            if hasattr(self, 'playwright') and self.playwright:
                self.playwright.stop()
        except Exception as e:
            print(f"Error closing Playwright: {e}") 