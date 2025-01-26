import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.parser.parser_class import Parser
from data_collection.downloader.downloader_class import Downloader

def test_main():
    years: list[int] = [2013]
    quartrs: list[int] = [3] 
    raw_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\2013_reports")

    parser = Parser(years, quartrs, raw_files_dir)
    company_links: dict[str, list[dict]] = parser.get_company_links_object()
    
    downloader = Downloader(company_links, raw_files_dir)
    downloader.download_files()

if __name__ == "__main__":
    test_main()