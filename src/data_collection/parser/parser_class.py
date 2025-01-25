import requests
import pandas as pd
import os
from pathlib import Path

class Parser:
    """
    This class is designed to parse and retrieve financial report data from the SEC's EDGAR database. 
    It downloads index file links for specified years and quarters, extracts relevant 10-K and 10-Q filings, 
    and filters them for S&P 500 companies. The class provides methods to compile these links into a 
    structured format for further usage.
    """
    
    headers = {
    'User-Agent': 'mvshibanov@edu.hse.ru'
}

    def __init__(self, years: list[int], quartrs: list[int], save_dir: Path):
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
        self.save_dir = save_dir
        self.all_lines: list = []
        self.reports_df = pd.DataFrame()
        self.loaded_comapines_set = set()
        self.snp_remainings_df = pd.DataFrame()
        self.snp_quarter_df = pd.DataFrame()
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
    
    def get_all_links(self):
        """
        Processes self.all_lines (each line from the SEC master index files), splits them, 
        and appends parsed entries into a DataFrame (self.reports_df). Each valid entry has
        5 fields: 'cik', 'name', 'type', 'filed_date', 'file'.

        Returns:
         - A pandas DataFrame (self.reports_df) containing the structured data from the parsed lines.
        """
        report_releas_lst = []

        for line in self.all_lines:
            part = line.split('|')
            if len(part) == 5:
                report_releas_lst.append(part)
        print(f'All links prepared. the len: {len(report_releas_lst)}')

        self.reports_df = pd.DataFrame(report_releas_lst, columns=['cik', 'name', 'type', 'filed_date', 'file'])

        return self.reports_df

    
    def check_remaining_companies(self) -> set:
        """
        Checks which companies have already been downloaded into self.save_dir by listing 
        the existing subdirectories in that folder. It then stores those in self.loaded_comapines_set.

        Returns:
         - A set of company identifiers (directory names) that are already present in self.save_dir.
        """
        downloaded_companies = []

        for company_dir in os.listdir(self.save_dir):
            downloaded_companies.append(company_dir)

        self.loaded_comapines_set = set(downloaded_companies)

        return self.loaded_comapines_set

    def get_snp_cik(self) -> pd.DataFrame:
        """
        Reads the S&P 500 companies list from Wikipedia, ensures each CIK (unique identifier)
        is only taken once, and filters out companies that have already been downloaded.
        The remaining companies are stored in self.snp_remainings_df.

        Returns:
         - A pandas DataFrame (self.snp_remainings_df) of remaining S&P 500 companies
           that have not been downloaded yet.
        """
        link = (
            #"https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        df = pd.read_html(link, header=0)[0]
        df = df.astype(str)

        df['Count'] = df.groupby('CIK').cumcount()
        qnique_cik = df[df['Count'] == 0]
        snp_str_df = qnique_cik.drop(columns=['Count'])

        mask = snp_str_df['Symbol'].apply(lambda x: x not in self.loaded_comapines_set)
        self.snp_remainings_df = snp_str_df[mask]

        return self.snp_remainings_df
    
    def get_snp_links(self) -> pd.DataFrame:
        """
        Filters self.reports_df to include only the filings of type '10-K' or '10-Q'.
        Then it matches these filings against the S&P 500 companies (self.snp_remainings_df) 
        by their CIK. The matched entries are stored in self.snp_quarter_df.

        Returns:
         - A pandas DataFrame (self.snp_quarter_df) containing 10-K/10-Q filings for
           the remaining S&P 500 companies.
        """
        self.reports_df = self.reports_df[self.reports_df['type'].isin(['10-K', '10-Q'])]

        self.snp_remainings_df['CIK'] = pd.to_numeric(self.snp_remainings_df['CIK']).astype(int)
        self.reports_df['cik'] = pd.to_numeric(self.reports_df['cik']).astype(self.snp_remainings_df['CIK'].dtype)

        self.snp_remainings_df.set_index('CIK', inplace=True)

        snp_quarter_df = self.reports_df.merge(
            self.snp_remainings_df[['Symbol']],
            left_on='cik',
            right_index=True,
            how='inner'
        )

        snp_quarter_df.rename(columns={'Symbol': 'ticker'}, inplace=True)
        self.snp_quarter_df = snp_quarter_df[['ticker', 'cik', 'name', 'type', 'filed_date', 'file']]

        print(f"DataFrame created:\n{self.snp_quarter_df}")
        return self.snp_quarter_df
    
    @staticmethod
    def get_links_n_dates(snp_quarter_df: pd.DataFrame, company_name: str) -> list[str]:
        """
        For a given company (specified by ticker), sorts its filings by 'filed_date' in 
        descending order. Returns a list of dictionaries containing the 'page_link' 
        (SEC filing path) and 'filed_date' for each record.

        Parameters:
         - snp_quarter_df (pd.DataFrame): DataFrame containing the relevant filings.
         - company_name (str): The company's ticker symbol.

        Returns:
         - A list of dictionaries, each holding 'page_link' and 'filed_date' for the company's filings.
        """
        links_n_dates = []

        subset_df = snp_quarter_df[snp_quarter_df['ticker'] == company_name].copy()
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
        Iterates over all remaining S&P 500 company tickers in self.snp_quarter_df,
        retrieves their sorted filings via get_links_n_dates, and compiles the results
        into a dictionary (self.company_links_object) where keys are tickers and 
        values are lists of filing links/dates.
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
        company_names = self.snp_quarter_df['ticker'].unique()

        for company_name in company_names:

            links_n_dates = Parser.get_links_n_dates(self.snp_quarter_df, company_name)

            self.company_links[company_name] = links_n_dates

    def get_company_links_object(self) -> dict[str, list[dict]]:
        """
        Orchestrates the entire process:
         - Iterates over self.years and self.quartrs to download the SEC index files.
         - Calls get_all_links() to structure the downloaded lines into self.reports_df.
         - Calls check_remaining_companies() to see which companies are already downloaded.
         - Calls get_snp_cik() to read the S&P 500 list and filter out those already downloaded.
         - Calls get_snp_links() to match the S&P 500 with the 10-K/10-Q filings.
         - Calls compile_links() to build a dictionary of company tickers mapped to 
           their respective filings and dates.

        Returns:
         - The compiled dictionary (self.company_links_object) with company tickers 
           mapped to their list of filing links and filing dates.
        """
        for year in self.years:
            for qrt in self.quartrs:

                lines = Parser.download_report(year, qrt)

                self.all_lines.extend(lines)
                print(year, qrt)
        
        self.get_all_links()

        self.check_remaining_companies()

        self.get_snp_cik()

        self.get_snp_links()

        self.compile_links()

        return self.company_links




    