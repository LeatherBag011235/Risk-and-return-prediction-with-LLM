import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

import time
import Path

from parser import Parser
from downloader import Downloader
from converter import Converter


start_time = time.time()

def main():
    data_dir: Path = Path(...)

    company_links = Parser.get_liks()

    downloader = Downloader(company_links)
    downloader.download

    Converter(data_dir)

if __name__ == "__main__":
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")