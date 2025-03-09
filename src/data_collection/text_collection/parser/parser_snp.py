from pathlib import Path
import pandas as pd

from data_collection.logging_config import logger
from data_collection.parser.parser_class import Parser

class ParserSnP(Parser):

    def __init__(self, years: list[int], quartrs: list[int], raw_files_dir: Path):
        super().__init__(years, quartrs, raw_files_dir)
        self.loaded_comapines_set = set()
        self.snp_remainings_df = pd.DataFrame()
      

    def check_remaining_companies(self) -> set:
        """
        Checks which companies have already been downloaded into self.save_dir by listing 
        the existing subdirectories in that folder. It then stores those in self.loaded_comapines_set.

        Returns:
         - A set of company identifiers (directory names) that are already present in self.save_dir.
        """
        downloaded_companies = []

        self.save_dir.mkdir(parents=True, exist_ok=True)
        for company_dir in self.save_dir.iterdir():
            downloaded_companies.append(company_dir.name)

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
        by their CIK. The matched entries are stored in self.final_df.

        Returns:
         - A pandas DataFrame (self.final_df) containing 10-K/10-Q filings for
           the remaining S&P 500 companies.
        """
        self.reports_df = self.reports_df[self.reports_df['type'].isin(['10-K', '10-Q'])]

        self.snp_remainings_df['CIK'] = pd.to_numeric(self.snp_remainings_df['CIK']).astype(int)
        self.reports_df['cik'] = pd.to_numeric(self.reports_df['cik']).astype(self.snp_remainings_df['CIK'].dtype)

        self.snp_remainings_df.set_index('CIK', inplace=True)

        final_df = self.reports_df.merge(
            self.snp_remainings_df[['Symbol']],
            left_on='cik',
            right_index=True,
            how='inner'
        )

        final_df.rename(columns={'Symbol': 'ticker'}, inplace=True)
        self.final_df = final_df[['ticker', 'cik', 'name', 'type', 'filed_date', 'file']]

        logger.info(f"DataFrame created:\n{self.final_df}")
        return self.final_df
    
    def get_company_links(self) -> dict[str, list[dict]]:
        """
        Orchestrates the entire process:
         - Calls get_all_links() to iterate over self.years and self.quartrs to download the SEC index files.
         - Calls get_all_links() to structure the downloaded lines into self.reports_df.
         - Calls check_remaining_companies() to see which companies are already downloaded.
         - Calls get_snp_cik() to read the S&P 500 list and filter out those already downloaded.
         - Calls get_snp_links() to match the S&P 500 with the 10-K/10-Q filings.
         - Calls compile_links() to build a dictionary of company tickers mapped to 
           their respective filings and dates.

        Returns:
         - The compiled dictionary (self.company_links) with company tickers 
           mapped to their list of filing links and filing dates.
        """
       
        self.get_all_lines()

        self.get_all_links()

        self.check_remaining_companies()

        self.get_snp_cik()

        self.get_snp_links()

        self.compile_links()

        return self.company_links