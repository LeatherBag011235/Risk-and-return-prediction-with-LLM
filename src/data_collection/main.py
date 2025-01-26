import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

import time
from pathlib import Path

from parser import Parser
from downloader import Downloader
from converter import Converter


start_time = time.time()

def main():
    years: list[int] = [2013, 2012]
    quartrs: list[int] = [3, 4] 

    raw_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\2013_reports")
    prepared_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\prepared_data\2013_reports")

    parser = Parser(years, quartrs, raw_files_dir)
    company_links: dict[str, list[dict]] = parser.get_company_links_object()
    
    downloader = Downloader(company_links, raw_files_dir)
    downloader.download_files()

    converter = DictionaryConverter(raw_files_dir, prepared_files_dir)
    converter.convert_files()

    Converter(save_dir)

if __name__ == "__main__":
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")