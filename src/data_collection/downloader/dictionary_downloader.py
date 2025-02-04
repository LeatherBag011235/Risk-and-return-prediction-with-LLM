from pathlib import Path
from bs4 import BeautifulSoup
import unidecode
import re
import polars as pl
import requests

from .downloader_class import Downloader
from data_collection.logging_config import logger

class DictionaryDownloader(Downloader):

    def __init__(self, company_links: dict[str, list[dict]], raw_files_dir: Path):
        super().__init__(company_links, raw_files_dir)

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
        logger.info(f"Text for {company_name} ==> {file_path} saved successfully.")

    def process_filing(self, session: requests.Session, company_name: str, filing: dict) -> None:
        """
        Process a single filing: download, clean, and save.

        Args:
            session (requests.Session): HTTP session for requests.
            company_name (str): Ticker of the company.
            filing (dict): Metadata for the filing.
        """
        soup = DictionaryDownloader.get_soup(session, company_name, filing['page_link'])
        if not soup:
            return

        soup = DictionaryDownloader.delete_tables(soup)
        text = DictionaryDownloader.text_preprocessing(soup)
        cleaned_text = DictionaryDownloader.clean_text(text)

        if len(cleaned_text) > 10000:
            self.save_file(cleaned_text, company_name, filing['filed_date'])
        else:
            logger.warning(f"{company_name} filing {filing['filed_date']} has less than 10,000 characters. Not saved.")