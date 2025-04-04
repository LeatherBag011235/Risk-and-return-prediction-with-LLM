import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import unidecode
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .consts import start_pattern, start_pattern_reserve, end_pattern
from src.data_collection.logging_config import logger

class Downloader(ABC):
    """
    Downloader class to scrape, process, and save financial reports from SEC archives.

    Attributes:
        headers (dict): Headers for HTTP requests. User-Agent must be defined.
        company_links (dict): Mapping of company tickers to their SEC filing liks and filed dates,
        sorted in decending order (most reasent first).
        save_dir (Path): Directory path where processed files will be saved.
    """
    headers = {
    'User-Agent': 'mvshibanov@edu.hse.ru'
}

    def __init__(self, company_links: dict[str, list[dict]]=None, raw_files_dir: Path=None):
        """
        Args:
            company_links (dict[str, list[dict]]): Structure containing company tickers and their filing metadata.
            save_dir (Path): Directory where processed files will be saved.

        Structure of company_links_object:
                {
            "AAPL": [
                {
                    "page_link": "/Archives/edgar/data/320193/000032019323000067/0000320193-23-000067-index.htm",
                    "filed_date": "2023-09-30"
                },
                {
                    "page_link": "/Archives/edgar/data/320193/000032019323000066/0000320193-23-000066-index.htm",
                    "filed_date": "2023-06-30"
                }
            ],
            "MSFT": [
                {
                    "page_link": "/Archives/edgar/data/789019/000078901923000012/0000789019-23-000012-index.htm",
                    "filed_date": "2023-08-15"
                },
                {
                    "page_link": "/Archives/edgar/data/789019/000078901923000011/0000789019-23-000011-index.htm",
                    "filed_date": "2023-05-15"
                }
            ]
        }

        """
        self.company_links = company_links
        self.save_dir = raw_files_dir

    @staticmethod
    def get_soup(session: requests.Session, company_name: str, url: str) -> BeautifulSoup:
        """
        Fetch and parse HTML content from SEC archives.

        Args:
            session (requests.Session): Shared HTTP session for making requests.
            company_name (str): Ticker of the company being processed.
            url (str): Partial URL to the SEC filing.

        Returns:
            BeautifulSoup: Parsed HTML content of the filing, or None if all attempts fail.
        """
        ATTEMPTS_AMOUNT = 3
        TIMEOUT = 15
        backoff_time = 2

        full_url = f'https://www.sec.gov/Archives/{url}'

        for attempt in range(ATTEMPTS_AMOUNT):
            try:
                response = session.get(full_url, headers=Downloader.headers, timeout=TIMEOUT)
                response.raise_for_status()
                content  = response.text

                start_match = start_pattern.search(content)
                if start_match:
                    start_idx = start_match.end()
                else:
                    logger.info(f"Failed to find start index in {company_name} ==> {full_url}. Using fallback pattern.")
                    start_match = start_pattern_reserve.search(content)

                    if start_match:
                        start_idx = start_match.end()
                    else:
                        logger.error(f"Failed to find start index with fallback in {company_name} ==> {full_url}")
                        return None

                end_match = end_pattern.search(content, start_idx)
                if end_match:
                    end_idx = end_match.end()
                else:
                    logger.error(f"Failed to find end index in {company_name} ==> {full_url}")
                    return None

                html_content = content[start_idx:end_idx]
                soup = BeautifulSoup(html_content, 'html.parser')
                logger.info(f"Page {company_name} ==> {full_url} parsed successfully.")
                return soup
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} of {ATTEMPTS_AMOUNT} failed for {full_url}: {e}\nRetrying after {backoff_time}...")
                time.sleep(backoff_time) 
                backoff_time *= 2

        logger.error(f"All attempts failed for {company_name} ==> {full_url}")
        return None
    
    @staticmethod
    def create_session():
        """
        Creates a requests.Session() with an increased connection pool limit.
        """
        session = requests.Session()
        
        # Configure the adapter to increase the pool size
        adapter = HTTPAdapter(
            pool_connections=35,  # Increase max connections
            pool_maxsize=35,  # Max simultaneous requests
            max_retries=Retry(total=5, backoff_factor=1.5)  # Retry failed requests
        )
        
        session.mount("https://", adapter)
        return session
    
    @staticmethod
    def text_preprocessing(soup: BeautifulSoup) -> str:
        """
        Extract and normalize text content from the HTML.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            str: Normalized text content.
        """
        text = soup.get_text()
        return unidecode.unidecode(text)
    
    @abstractmethod
    def save_file(self):
        """
        Save prepared text as.
        """
        pass
    
    @abstractmethod
    def process_filing(self, session: requests.Session, company_name: str, filing: dict) -> None:
        """
        Process a single filing: download, clean, and save.
        """
        pass
    
    def download_files(self) -> None:
        """
        Download and process all files using parallel processing.
        """
        session = requests.Session()
        all_tasks = []

        with ThreadPoolExecutor() as executor:
            for company_name, filings in self.company_links.items():
                for filing in filings:
                    task = executor.submit(self.process_filing, session, company_name, filing)
                    all_tasks.append(task)

            for future in as_completed(all_tasks):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing a filing: {e}")

        logger.info("All filings processed.")
