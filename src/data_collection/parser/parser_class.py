import requests
import pandas as pd
import os

class Parser:

    headers = {
    'User-Agent': 'mvshibanov@edu.hse.ru'
}

    def __init__(self, years: list[int], quartrs: list[int]):
        self.years = years
        self.quartrs = quartrs
        self.all_lines: list = []
        self.reports_df = pd.DataFrame()
        self.loaded_comapines_set = set()
        self.snp_remainings_df = pd.DataFrame()
        self.snp_quarter_df = pd.DataFrame()
        self.company_links_object = {}


    @staticmethod
    def download_report(year: list[int], qtr: list[int]) -> list[str]:
        base_url = "https://www.sec.gov/Archives/edgar/full-index"
        index_url = f"{base_url}/{year}/QTR{qtr}/master.idx"

        response = requests.get(index_url, headers=Parser.headers)
        lines = response.text.splitlines()

        return lines
    
    def get_all_links(self):
        report_releas_lst = []

        for line in self.all_lines:
            part = line.split('|')
            if len(part) == 5:
                report_releas_lst.append(part)
        print(f'All links prepared. the len: {len(report_releas_lst)}')

        self.reports_df = pd.DataFrame(report_releas_lst, columns=['cik', 'name', 'type', 'filed_date', 'file'])

        return self.reports_df

    
    def check_remaining_companies(self) -> set:
        dir_to_check = r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\2013_reports"

        downloaded_companies = []

        for company_dir in os.listdir(dir_to_check):
            downloaded_companies.append(company_dir)

        self.loaded_comapines_set = set(downloaded_companies)

        return self.loaded_comapines_set

    def get_snp_cik(self) -> pd.DataFrame:
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

    def compile_links(self) -> dict:
        company_names = self.snp_quarter_df['ticker'].unique()

        for company_name in company_names:

            links_n_dates = Parser.get_links_n_dates(self.snp_quarter_df, company_name)

            self.company_links_object[company_name] = links_n_dates

    def get_company_links_object(self):
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

        return self.company_links_object




    