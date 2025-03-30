import os
import time
import json
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 검색어 리스트
search_terms = ["banana"]

class BingImageScraper:
    def __init__(self, base_dir="selenium_images", max_images=100):
        self.base_dir = base_dir
        self.max_images = max_images
        self.session = self._create_request_session()

        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)

    def _create_request_session(self):
        """Create a requests session with retry capabilities"""
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        })
        return session



    def setup_driver(self):
        """Initialize and configure the Selenium webdriver"""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-usb")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--remote-debugging-port=8005")
        options.add_argument("--max-memory-restart=500M")  # 메모리 초과 시 재시작
        # Anti-bot detection settings
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # Additional anti-bot measures
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
        })
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        return driver

    def download_image(self, url, path):
        """Download image from URL and save to path"""
        try:
            img_data = self.session.get(url, timeout=10).content
            with open(path, "wb") as f:
                f.write(img_data)
            return True
        except Exception as e:
            logger.error(f"다운로드 실패 ({url}): {e}")
            return False

    def download_images_parallel(self, image_data):
        """Download multiple images in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(
                lambda x: self.download_image(x[0], x[1]),
                image_data
            ))
        return sum(results)  # Return count of successful downloads

    def extract_image_url_from_data(self, element):
        """Extract image URL from Bing's data attribute"""
        try:
            parent = element.find_element(By.XPATH, "./ancestor::a") if element.tag_name != "a" else element
            data_json = parent.get_attribute("m")

            if data_json:
                data = json.loads(data_json)
                if "murl" in data:
                    return data["murl"]
        except Exception as e:
            logger.debug(f"데이터 속성에서 URL 추출 실패: {e}")
        return None

    def extract_image_url_from_viewer(self, driver, wait):
        """Extract image URL from the image viewer after clicking a thumbnail"""
        image_selectors = [
            ".mainImage",
            "#mainImageWindow img",
            ".mimg"
        ]

        for selector in image_selectors:
            try:
                image = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                src = image.get_attribute("src")
                if src and src.startswith("http") and not src.startswith("data:image"):
                    return src
            except:
                continue
        return None

    def scrape_term(self, term):
        """Scrape images for a specific search term"""
        term_dir = os.path.join(self.base_dir, "_"+term.replace("object only ", ""))
        if os.path.exists(term_dir) and os.listdir(term_dir):  # 폴더가 있고 파일이 있을 경우
            logger.info(f"'{term}': 이미 폴더가 존재하여 스킵")
            return 0
        os.makedirs(term_dir, exist_ok=True)

        driver = self.setup_driver()
        wait = WebDriverWait(driver, 10)

        try:
            # Navigate to Bing image search
            query = f"{term}"
            url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2&first=1"
            driver.get(url)
            time.sleep(3)  # Short wait for page load

            # Take screenshot for debugging
            driver.save_screenshot(os.path.join(self.base_dir, f"{term}_screenshot.png"))

            # Scroll to load more images
            for _ in range(5):  # Increased scroll count for more images
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)

            # Find thumbnails
            thumbnail_selectors = [".mimg", ".iusc img", "[class*='mimg']"]
            thumbnails = []

            for selector in thumbnail_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        thumbnails = elements
                        logger.info(f"'{term}': 셀렉터 '{selector}'로 {len(thumbnails)}개 썸네일 발견")
                        break
                except Exception as e:
                    logger.debug(f"셀렉터 '{selector}' 시도 중 오류: {e}")

            # Collect image URLs directly from data attributes when possible
            image_data = []  # List of (url, path) tuples for parallel download
            count = 0

            # First attempt: Extract URLs directly from data attributes (faster)
            for i, thumb in enumerate(thumbnails):
                if count >= self.max_images:
                    break

                try:
                    # Try to extract URL from data attribute first (no clicking needed)
                    src = self.extract_image_url_from_data(thumb)

                    if src:
                        file_path = os.path.join(term_dir, f"{term}_{count}.jpg")
                        image_data.append((src, file_path))
                        count += 1

                        if count % 10 == 0:
                            logger.info(f"{term}: {count} URLs collected")
                except Exception as e:
                    logger.debug(f"Error processing thumbnail {i}: {e}")

            # If we still need more images, try the click-and-extract method
            if count < self.max_images and thumbnails:
                for i, thumb in enumerate(thumbnails):
                    if count >= self.max_images:
                        break

                    try:
                        # Scroll to thumbnail
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", thumb)
                        time.sleep(0.5)

                        # Click thumbnail
                        driver.execute_script("arguments[0].click();", thumb)
                        time.sleep(1.5)

                        # Extract URL from viewer
                        src = self.extract_image_url_from_viewer(driver, wait)

                        if src:
                            file_path = os.path.join(term_dir, f"{term}_{count}.jpg")
                            image_data.append((src, file_path))
                            count += 1

                        # Close viewer
                        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        time.sleep(0.5)
                    except Exception as e:
                        logger.debug(f"Error on click method for thumbnail {i}: {e}")
                        try:
                            ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        except:
                            pass

            # Download collected images in parallel
            logger.info(f"{term}: 수집된 {len(image_data)}개 URL 다운로드 시작")
            successful_downloads = self.download_images_parallel(image_data)
            logger.info(f"{term}: {successful_downloads}개 이미지 다운로드 완료")

            return successful_downloads

        except Exception as e:
            logger.error(f"{term} 처리 중 오류 발생: {e}")
            return 0
        finally:
            driver.quit()

    def run(self, terms=None):
        """Run the scraper on multiple search terms"""
        if terms is None:
            terms = search_terms

        results = {}
        for index, term in enumerate(terms):
            # if 1 < index:
            #     break
            logger.info(f"'{term}' 검색어 처리 시작({index}/{len(terms)})")
            count = self.scrape_term(term)
            results[term] = count
            logger.info(f"'{term}' 처리 완료: {count}개 이미지 다운로드")
            time.sleep(2)  # Small delay between terms

        logger.info("모든 작업 완료")
        return results


if __name__ == "__main__":
    scraper = BingImageScraper(max_images=100)
    scraper.run()