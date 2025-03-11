import sys 
sys.path.append(r'C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src')

from pathlib import Path

from data_collection.text_collection.parser.parser_snp import ParserSnP

def test_main():
    years: list[int] = [2023, ]
    quartrs: list[int] = [3, ] 
    raw_files_dir: Path = Path(r"C:\Users\310\Desktop\Progects_Py\data\Parsim_sec_data\raw_data\2013_reports")

    parser = ParserSnP(years, quartrs, raw_files_dir)
    
    print(len(parser.get_company_links()))
    print(parser.check_remaining_companies())

if __name__ == "__main__":
    test_main()