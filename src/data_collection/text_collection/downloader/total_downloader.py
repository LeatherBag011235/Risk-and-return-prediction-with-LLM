import datetime
import requests
import psycopg2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .downloader_class import Downloader
from data_collection.logging_config import logger

class TotalDownloader(Downloader):

    def __init__(self, db_params: dict,):
        """
        Initializes the ReportDownloader with database credentials.
        
        Args:
            db_params (dict): Database connection parameters.
        """
        self.db_params = db_params
        self.pending_reports = []

    def get_pending_reports(self) -> list[tuple[str, datetime.date, str]]: 
        """Fetches reports that haven't been downloaded yet.

        Returns:
            list[tuple[str, datetime, str]]: List of (CIK, filed_date, URL) tuples.
        """
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            cursor.execute("SELECT cik, filed_date, url FROM reports WHERE raw_text IS NULL;")
            self.pending_reports = cursor.fetchall()  # List of (cik: str, filed_date: datetime, url: str)

            cursor.close()
            conn.close()
            return self.pending_reports

        except Exception as e:
            print(f"❌ Database error: {e}")
            return []

    def save_file(self, text: str, cik: str, date: datetime) -> None:
        """
        Saves the downloaded financial report text into the PostgreSQL database.

        Args:
            text (str): The extracted text from the SEC filing.
            cik (str): CIK number of the company.
            date (datetime): Filing date in the form of datetime.date(yyyy, mm, dd).
        """
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE reports
                SET raw_text = %s
                WHERE cik = %s AND filed_date = %s;
            """, (text, cik, date))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"✅ Stored text for CIK {cik}, filed {date}")

        except Exception as e:
            logger.error(f"❌ Database error while storing text: {e}")

    def process_filing(self, session: requests.Session, cik: str, date: datetime, filing: str,) -> None:
        """
        Process a single filing: download, clean, and save.

        Args:
            session (requests.Session): HTTP session for requests.
            cik (str): CIK number of the company.
            date (datetime): date of filling this doc in the form of datetime.date(yyyy, mm, dd)
            filing (str): Part of url link to particular filling.
        """
        soup = TotalDownloader.get_soup(session, cik, filing)
        if not soup:
            return
        
        text = TotalDownloader.text_preprocessing(soup)

        if len(text) > 10000:
            self.save_file(text, cik, date)
        else:
            logger.warning(f"{cik} filing {date} has less than 10,000 characters. Not saved.")

    def download_files(self) -> None:
        """
        Download and process all files using parallel processing.
        """
        self.get_pending_reports()
        total_reports = len(self.pending_reports)
        session = TotalDownloader.create_session()

        with ThreadPoolExecutor(max_workers=32) as executor, tqdm(total=total_reports, desc="Downloading Reports", unit="file") as progress:
            futures = {
                executor.submit(self.process_filing, session, cik, date, filing): (cik, date) 
                for cik, date, filing in self.pending_reports
                }
                
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing a filing: {e}")
                
                progress.update(1)  

        logger.info("All filings processed.")
        