import sys 
sys.path.append(r'C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.downloader.total_downloader import TotalDownloader

DB_PARAMS = {
    "dbname": "reports_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

def test_main():
    downloader = TotalDownloader(DB_PARAMS)
    downloader.download_files() 

if __name__ == "__main__":
    test_main()