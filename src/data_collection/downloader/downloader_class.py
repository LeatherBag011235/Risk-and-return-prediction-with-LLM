from .consts import start_pattern, steart_pattern_reserve, end_pattern

import requests
from bs4 import BeautifulSoup
import time
import unidecode
import re
import polars as pl
import os

class Downloader:

    headers = {
    'User-Agent': 'mvshibanov@edu.hse.ru'
}

    def __init__(self, company_links: dict[str]):
        """
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

    @staticmethod
    def get_soup(session, company_name, url):
        ATTEMPTS_AMOUNT = 3
        TIMEOUT=15
        backoff_time = 2

        full_url = f'https://www.sec.gov/Archives/{url}'

        for attempt in range(ATTEMPTS_AMOUNT):
            try:
                response = session.get(full_url, headers=Downloader.headers, timeout=TIMEOUT)

                content  = response.text

                start_match = start_pattern.search(content)
                if start_match:
                    start_idx = start_match.end()
                else:
                    print(f'Fail to find start idx in {company_name} ==> {full_url} \n use <html> instead')
                    start_match = steart_pattern_reserve.search(content)

                    if start_match:
                        start_idx = start_match.end()
                    else:
                        print(f'Fail to find start idx with <html> {company_name} ==> {full_url} \n !!!!!!!!!!!!')

                end_match = end_pattern.search(content, start_idx)
                if end_match:
                    end_idx = end_match.end()
                else:
                    print(f'Fail to find end idx in{company_name} ==> {full_url}')

                html_content = content[start_idx:end_idx]
                soup = BeautifulSoup(html_content, 'html.parser')

                print(f"Page {company_name} ==> {full_url} paresed successfully.")

                return soup
            
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} of {ATTEMPTS_AMOUNT} failed for {full_url} retrying after {backoff_time} seconds...")
                time.sleep(backoff_time) 
                backoff_time *= 2

        print(f"All attempts failed for {company_name} ==> {full_url}")
        return None
    
    @staticmethod
    def delete_tabeles(soup):
        for script in soup(["script", "style", "head", "title", "meta", "[document]"]):
                script.decompose() 

        for table in soup.find_all('table'):
            table.decompose()

        for a in soup.find_all("a"):
            a.decompose()
        return soup
    
    @staticmethod
    def text_preprocessing(soup):
        text = soup.get_text()
        text = unidecode.unidecode(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        text = re.sub(r'\b\S*?\d\S*\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text 
    
    @staticmethod
    def save_file(text, company_name, filed_date):
        file_name = f'{filed_date}.parquet'
    
        text_col = pl.DataFrame(text.split())

        directory_path = f'./raw_data/full_snp_five_hundred/{company_name}/'

        if not os.path.exists(f'{directory_path}'):
            os.makedirs(f'{directory_path}')

        text_col.write_parquet(os.path.join(directory_path, file_name))

        print(f"Text for {company_name} ==> {os.path.join(directory_path, file_name)} saved successfully.")


    def download_files(self):
        fails = 0
        len_list = []
        all_docs_procesed = 0 

        session = requests.Session()

        for key in self.company_links_object.keys():

            for item_obj in self.company_links_object[key]:

                all_docs_procesed +=1

                soup1 = Downloader.get_soup(session, key, item_obj['page_link'])
                soup2 = Downloader.delete_tabeles(soup1)

                text = Downloader.text_preprocessing(soup2)

                cleaned_text = Downloader.clean_text(text)

                if len(cleaned_text) > 10000:

                    print(key, item_obj['filed_date'])

                    Downloader.save_file(cleaned_text, key, item_obj['filed_date'])
                    len_list.append(len(cleaned_text))

                else:
                    print(f" \n {key} \n len of {item_obj['page_link']} less then 10000 characters: {len(cleaned_text)} \n It is not saved \n")  

        print(f'Fails over all docs procesed: {fails/all_docs_procesed}')
        print(f'Average len of text {sum(len_list)/all_docs_procesed}')
