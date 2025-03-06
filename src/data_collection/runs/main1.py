import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

import time
from pathlib import Path

from data_collection.parser.parser_snp import ParserSnP
from data_collection.downloader.dictionary_downloader import DictionaryDownloader
from data_collection.converter.dictionary_converter import DictionaryConverter


start_time = time.time()

def main():
    years: list[int] = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    quartrs: list[int] = [1, 2, 3, 4] 

    raw_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\2009_till_2018_reports")
    prepared_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\prepared_data\2009_till_2018_reports")

    parser = ParserSnP(years, quartrs, raw_files_dir)
    company_links: dict[str, list[dict]] = parser.get_company_links ()
    
    downloader = DictionaryDownloader(company_links, raw_files_dir)
    downloader.download_files()

    converter = DictionaryConverter(raw_files_dir, prepared_files_dir)
    converter.convert_files()

if __name__ == "__main__":
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")