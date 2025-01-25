import requests
from bs4 import BeautifulSoup
import time
import unidecode
import re
import polars as pl
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .consts import start_pattern, steart_pattern_reserve, end_pattern

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class Downloader:
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

    def __init__(self, company_links: dict[str, list[dict]], save_dir: Path):
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
        self.save_dir = save_dir


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
                    logging.warning(f"Failed to find start index in {company_name} ==> {full_url}. Using fallback pattern.")
                    start_match = steart_pattern_reserve.search(content)

                    if start_match:
                        start_idx = start_match.end()
                    else:
                        logging.error(f"Failed to find start index with fallback in {company_name} ==> {full_url}")
                        return None

                end_match = end_pattern.search(content, start_idx)
                if end_match:
                    end_idx = end_match.end()
                else:
                    logging.error(f"Failed to find end index in {company_name} ==> {full_url}")
                    return None

                html_content = content[start_idx:end_idx]
                soup = BeautifulSoup(html_content, 'html.parser')
                logging.info(f"Page {company_name} ==> {full_url} parsed successfully.")
                return soup
            
            except requests.exceptions.RequestException as e:
                logging.warning(f"Attempt {attempt + 1} of {ATTEMPTS_AMOUNT} failed for {full_url}: {e}\nRetrying after {backoff_time}...")
                time.sleep(backoff_time) 
                backoff_time *= 2

        logging.error(f"All attempts failed for {company_name} ==> {full_url}")
        return None

    
    @staticmethod
    def delete_tables(soup: BeautifulSoup) -> BeautifulSoup:
        """
        Remove unnecessary elements (scripts, tables, links) from the HTML content.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            BeautifulSoup: Cleaned HTML content.
        """
        for script in soup(["script", "style", "head", "title", "meta", "[document]"]):
                script.decompose() 

        for table in soup.find_all('table'):
            table.decompose()

        for a in soup.find_all("a"):
            a.decompose()
        return soup
    

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
    

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Perform additional cleaning of text content.

        Args:
            text (str): Raw text content.

        Returns:
            str: Cleaned text with special characters and numbers removed.
        """
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        text = re.sub(r'\b\S*?\d\S*\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    

    def save_file(self, text: str, company_name: str, filed_date: str) -> None:
        """
        Save cleaned text as a Parquet file.

        Args:
            text (str): Cleaned text content.
            company_name (str): Ticker of the company.
            filed_date (str): Filing date, used as the filename.
        """
        file_name = f"{filed_date}.parquet"
        text_col = pl.DataFrame(text.split())

        directory_path = Path(self.save_dir) / company_name
        directory_path.mkdir(parents=True, exist_ok=True)

        file_path = directory_path / file_name
        text_col.write_parquet(file_path)
        logging.info(f"Text for {company_name} ==> {file_path} saved successfully.")


    def process_filing(self, session: requests.Session, company_name: str, filing: dict) -> None:
        """
        Process a single filing: download, clean, and save.

        Args:
            session (requests.Session): HTTP session for requests.
            company_name (str): Ticker of the company.
            filing (dict): Metadata for the filing.
        """
        soup = Downloader.get_soup(session, company_name, filing['page_link'])
        if not soup:
            return

        soup = Downloader.delete_tables(soup)
        text = Downloader.text_preprocessing(soup)
        cleaned_text = Downloader.clean_text(text)

        if len(cleaned_text) > 10000:
            self.save_file(cleaned_text, company_name, filing['filed_date'])
        else:
            logging.warning(f"{company_name} filing {filing['filed_date']} has less than 10,000 characters. Not saved.")

    
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
                    logging.error(f"Error processing a filing: {e}")

        logging.info("All filings processed.")
