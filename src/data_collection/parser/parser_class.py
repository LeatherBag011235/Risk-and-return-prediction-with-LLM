import requests
import pandas as pd
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from data_collection.logging_config import logger

class Parser(ABC):
    """
    This class is designed to parse and retrieve financial report data from the SEC's EDGAR database. 
    It downloads index file links for specified years and quarters, extracts relevant filings. 
    The class provides methods to compile these links into a 
    structured format for further usage.

    This is a parent abstract class so it provide basic functionality to get fillings from EDGAR databese.
    It suppose to be extended based on what particular companies and fillings do you want to parse.
    """
    
    headers = {
    'User-Agent': 'mvshibanov@edu.hse.ru'
}

    def __init__(self, years: list[int], quartrs: list[int], raw_files_dir: Path=None):
        """
        Initializes the Parser instance.

        Variables:
         - years (list[int]): A list of years (e.g. [2013, 2014]) to parse.
         - quartrs (list[int]): A list of quarter numbers (e.g. [1, 2, 3, 4]) to parse.
         - save_dir (Path): The directory path where data (company directories/files) will be stored.

        Where They Are Used:
         - self.years and self.quartrs are used in get_company_links_object() to download
           the SEC index files for each specified year and quarter.
         - self.save_dir is used in check_remaining_companies() to identify which companies
           have already been downloaded and stored in that directory.
        """
        self.years = years
        self.quartrs = quartrs
        self.save_dir = raw_files_dir
        self.all_lines: list = []
        self.reports_df = pd.DataFrame()
        self.final_df = pd.DataFrame()
        self.company_links = {}


    @staticmethod
    def download_report(year: list[int], qtr: list[int]) -> list[str]:
        """
        Downloads the SEC master index file for the given year and quarter, then returns 
        the file content split by lines.

        Parameters:
         - year (list[int]): The year for which to download the index file.
         - qtr (list[int]): The quarter for which to download the index file.

        Returns:
         - A list of strings, each representing a line from the downloaded master index file.
        """
        base_url = "https://www.sec.gov/Archives/edgar/full-index"
        index_url = f"{base_url}/{year}/QTR{qtr}/master.idx"

        response = requests.get(index_url, headers=Parser.headers)
        lines = response.text.splitlines()

        return lines
    
    def get_all_lines(self) -> list[str]:
        for year in self.years:
            for qrt in self.quartrs:

                lines = Parser.download_report(year, qrt)

                self.all_lines.extend(lines)
                logger.info(f'{year} {qrt}')

        return self.all_lines
    
    def get_all_links(self):
        """
        Processes self.all_lines (each line from the SEC master index files), splits them, 
        and appends parsed entries into a DataFrame (self.reports_df). Each valid entry has
        5 fields: 'cik', 'name', 'type', 'filed_date', 'file'.

        Returns:
         - A pandas DataFrame (self.reports_df) containing the structured data from the parsed lines.
         It is all links for all fillings for the specified period.
        """
        report_releas_lst = []
 
        for line in self.all_lines:
            part = line.split('|')
            if len(part) == 5:
                report_releas_lst.append(part)
        logger.info(f'All links prepared. the len: {len(report_releas_lst)}')

        self.reports_df = pd.DataFrame(report_releas_lst, columns=['cik', 'name', 'type', 'filed_date', 'file'])

        return self.reports_df

    @abstractmethod
    def check_remaining_companies(self) -> set:
        """
        Checks which companies have already been downloaded into self.save_dir by listing 
        the existing subdirectories in that folder. It then stores those in self.loaded_comapines_set.

        Returns:
         - A set of company identifiers (directory names) that are already present in self.save_dir.
        """
        pass

    
    @staticmethod
    def get_links_n_dates(final_df: pd.DataFrame, company_name: str) -> list[str]:
        """
        For a given company (specified by ticker), sorts its filings by 'filed_date' in 
        descending order. Returns a list of dictionaries containing the 'page_link' 
        (SEC filing path) and 'filed_date' for each record.

        Parameters:
         - final_df (pd.DataFrame): DataFrame containing the relevant filings.
         - company_name (str): The company's ticker symbol.

        Returns:
         - A list of dictionaries, each holding 'page_link' and 'filed_date' for the company's filings.
        """
        links_n_dates = []

        subset_df = final_df[final_df['ticker'] == company_name].copy()
        subset_df['filed_date'] = pd.to_datetime(subset_df['filed_date'], format='%Y-%m-%d')

        sorted_df = subset_df.sort_values(by='filed_date', ascending=False)
        sorted_df["filed_date"] = sorted_df["filed_date"].astype(str)

        for row in sorted_df.itertuples():
            row_dict = {}

            row_dict['page_link'] = row.file
            row_dict['filed_date'] = row.filed_date

            links_n_dates.append(row_dict)

        return links_n_dates
    

    def compile_links(self) -> dict[str, list[dict]]:
        """
        Iterates over all remaining company tickers in self.final_df,
        retrieves their sorted filings via get_links_n_dates, and compiles the results
        into a dictionary (self.company_links) where keys are tickers and 
        values are lists of filing links/dates.
        Structure of company_links:
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
        company_names = self.final_df['ticker'].unique()

        for company_name in company_names:

            links_n_dates = Parser.get_links_n_dates(self.final_df, company_name)

            self.company_links[company_name] = links_n_dates


    @abstractmethod
    def get_company_links(self) -> dict[str, list[dict]]:
        """
        Orchestrates the entire process...
         
        Returns:
         - The compiled dictionary (self.company_links_object) with company tickers 
           mapped to their list of filing links and filing dates.
        """
        pass