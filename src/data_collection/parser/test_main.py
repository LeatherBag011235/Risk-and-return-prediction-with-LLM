import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

from data_collection.parser.parser_class import Parser

def test_main():
    years: list[int] = [2022,2023]
    quartrs: list[int] = [3, 4] 

    parser = Parser(years, quartrs)
    parser.get_company_links_object()

if __name__ == "__main__":
    test_main()