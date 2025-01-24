import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.parser.parser_class import Parser

def test_main():
    years: list[int] = [2023, 2022]
    quartrs: list[int] = [3, 4] 
    save_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\2013_reports")

    parser = Parser(years, quartrs, save_dir)
    print(parser.get_company_links_object())

if __name__ == "__main__":
    test_main()