from pathlib import Path
import pandas as pd
import requests
import psycopg2

from data_collection.parser.parser_class import Parser  
from data_collection.logging_config import logger  

class ParserTotal(Parser):
    """
    This class extends the Parser class to specifically fetch SEC reports (10-Q, 10-K)
    and store them in a PostgreSQL database.
    """

    def __init__(self, years: list[int], quartrs: list[int], db_params: dict,):
        """
        Initializes ParserTotal with database connection details.

        Parameters:
        - years: List of years to fetch SEC data for.
        - quartrs: List of quarters (1-4) to fetch SEC data for.
        - db_params: Dictionary containing database credentials.
        - raw_files_dir: is not needed here.
        """
        super().__init__(years, quartrs,)
        self.db_params = db_params  

    def filter_type(self)-> pd.DataFrame:
        """
        Filters `self.reports_df` to keep only '10-Q' and '10-K' filings.
        """
        self.reports_df = self.reports_df[self.reports_df["type"].isin(["10-Q", "10-K"])]
        return self.reports_df

    def check_remaining_companies(self) -> pd.DataFrame:
        """
        Retrieves existing (cik, filed_date) entries from the database.

        Returns:
        - A Pandas DataFrame with two columns: ['cik', 'filed_date'] containing existing entries.
        """
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            # Fetch stored (cik, filed_date)
            cursor.execute("SELECT cik, filed_date FROM reports;")
            existing_entries = cursor.fetchall()  # Example: [('0001234', '2023-03-01'), ...]

            cursor.close()
            conn.close()

            # Convert to DataFrame
            existing_df = pd.DataFrame(existing_entries, columns=['cik', 'filed_date'])

            existing_df['cik'] = existing_df['cik'].astype(str)
            existing_df['filed_date'] = existing_df['filed_date'].astype(str)

            logger.info(f"✅ Found {len(existing_df)} existing reports in DB.")
            return existing_df

        except Exception as e:
            logger.error(f"❌ Database error in `get_existing_entries()`: {e}")
            return pd.DataFrame(columns=['cik', 'filed_date'])  
        
    def filter_existing_entries(self) -> pd.DataFrame:
        """
        Filters `self.reports_df` by removing rows that already exist in the database using Pandas merge.

        This method modifies `self.reports_df` **in-place**.
        """
        try:
            existing_df = self.check_remaining_companies()  # Get existing (cik, filed_date)

            if existing_df.empty:
                logger.info("✅ No existing reports found, keeping all entries.")
                return 
            
            before_count = len(self.reports_df)
           
            self.reports_df['cik'] = self.reports_df['cik'].astype(str)
            self.reports_df['filed_date'] = self.reports_df['filed_date'].astype(str)

            self.reports_df = self.reports_df.merge(
                existing_df, 
                on=['cik', 'filed_date'], 
                how='left', 
                indicator=True)

            # Keep only new reports
            self.reports_df = self.reports_df[self.reports_df['_merge'] == 'left_only'].drop(columns=['_merge'])
            after_count = len(self.reports_df)

            logger.info(f"✅ Filtered reports. Kept {after_count} new entries (removed {before_count - after_count}).")
            return self.reports_df
        
        except Exception as e:
            logger.error(f"❌ Error in `filter_existing_entries()`: {e}")

    def get_cik_ticker_mapping(self) -> pd.DataFrame:
        """
        Retrieves and merges the CIK-to-Ticker mapping from the SEC's JSON dataset.

        This method:
        1. Downloads the **SEC CIK-Ticker JSON dataset**.
        2. Converts it into a **Pandas DataFrame**.
        3. Renames and **ensures correct formatting of CIKs**.
        4. Performs an **INNER JOIN** with `self.reports_df` to match CIKs with tickers.
        5. Logs debugging information, including dataset lengths and NaN values in tickers.

        Returns:
            pd.DataFrame: A filtered DataFrame containing:
                - 'cik' (Company Identifier)
                - 'ticker' (Stock ticker, if available)
                - 'filed_date' (Date of the SEC filing)

        Raises:
            requests.RequestException: If the SEC JSON request fails.
            ValueError: If the response cannot be parsed into a DataFrame.
        """
        response = requests.get(self.SEC_CIK_TICKER_URL, headers=self.headers)
        data = response.json()
        
        cik_ticker_df = pd.DataFrame.from_dict(data, orient="index")
        cik_ticker_df = cik_ticker_df.rename(columns={'cik_str': 'cik'})
        cik_ticker_df['cik'] = cik_ticker_df['cik'].astype(str)
        
        logger.debug(f"len of repors_df: {len(self.reports_df)}")
        logger.debug(f"len of cik_ticker_df: {len(cik_ticker_df)}")

        self.reports_df = self.reports_df.merge(
            cik_ticker_df, 
            on="cik", 
            how="inner"
            )
        
        logger.debug(f"NANS: {self.reports_df["ticker"].isna().sum()}")

        return self.reports_df[['cik', 'ticker', 'filed_date']]
    
    def store_reports_in_db(self) -> None:
        """
        Stores parsed SEC filings into the PostgreSQL database.
        """
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            for _, row in self.reports_df.iterrows():
                cik, name, report_type, filed_date, file_url, ticker, title = row
                
                # Ensure the company exists in the company tabel in DB
                cursor.execute("""
                    INSERT INTO companies (cik, name, ticker) 
                    VALUES (%s, %s, %s) 
                    ON CONFLICT (cik) DO NOTHING;
                """, (cik, name, ticker))

                # Insert the report info in reports tabel in DB
                cursor.execute("""
                    INSERT INTO reports (cik, filed_date, report_type, url, raw_text)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (cik, filed_date) DO NOTHING;
                """, (cik, filed_date, report_type, file_url, None))  # No raw_text yet

            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ Reports successfully stored in database.")

        except Exception as e:
            logger.error(f"❌ Database error: {e}")

    def get_company_links(self) -> None:
        """
        Fetches all SEC filing links and processes them. Download meta data about SEC fillings into db
        
        """
        self.get_all_lines()            # Fetch all raw SEC lines
        self.get_all_links()            # Process them into a DataFrame
        self.filter_type()              # Filters out all filings of the wrong type from self.reports_df
        self.filter_existing_entries()  # Filter out all filings that are already in database
        self.get_cik_ticker_mapping()   # Matches cik numbers and ticker from a separate SEC page
        self.store_reports_in_db()    
        

    

