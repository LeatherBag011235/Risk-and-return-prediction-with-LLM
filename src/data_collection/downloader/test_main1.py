import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.parser.parser_snp import ParserSnP
from data_collection.downloader.dictionary_downloader import DictionaryDownloader

def test_main():
    years: list[int] = [2013]
    quartrs: list[int] = [3] 
    raw_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\test_dir")

    parser = ParserSnP(years, quartrs, raw_files_dir)
    company_links: dict[str, list[dict]] = parser.get_company_links()
    
    downloader = DictionaryDownloader(company_links, raw_files_dir)
    downloader.download_files()

if __name__ == "__main__":
    test_main()