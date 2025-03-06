from pathlib import Path
import pandas as pd

import psycopg2
import pandas as pd
from pathlib import Path

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
        - raw_files_dir: Optional path to store raw SEC reports.
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

    def store_reports_in_db(self) -> None:
        """
        Stores parsed SEC filings into the PostgreSQL database.
        """
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()

            for _, row in self.reports_df.iterrows():
                cik, name, report_type, filed_date, file_url = row
                
                # Ensure the company exists in the company tabel in DB
                cursor.execute("""
                    INSERT INTO companies (cik, name) 
                    VALUES (%s, %s) 
                    ON CONFLICT (cik) DO NOTHING;
                """, (cik, name))

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
        Fetches all SEC filing links and processes them.
        Returns:
        - Dictionary mapping company tickers to their filing URLs.
        """

        self.get_all_lines()           # Fetch all raw SEC lines
        self.get_all_links()           # Process them into a DataFrame
        self.filter_type()             # Filters out all filings of the wrong type from self.reports_df
        self.check_remaining_companies()
        self.filter_existing_entries() # Filter out all filings that are already in database
        self.store_reports_in_db()    
        

    

