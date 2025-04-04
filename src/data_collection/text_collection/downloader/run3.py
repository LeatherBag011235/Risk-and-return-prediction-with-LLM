import pandas as pd

from src.data_collection.text_collection.downloader.total_downloader import TotalDownloader

DB_PARAMS = {
    "dbname": "reports_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

link = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(link, header=0)[0].astype(str)
snp_ciks: list[str] = df["CIK"].to_list()

def main():
    downloader = TotalDownloader(DB_PARAMS)
    downloader.download_files(cik_list=snp_ciks) 

if __name__ == "__main__":
    main()