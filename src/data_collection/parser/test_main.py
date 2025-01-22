import sys 
sys.path.append(r'C:\Users\310\Desktop\Progects_Py\Risk-and-return-prediction-with-LLM\src')

from data_collection.parser.parser_class import Parser

def main():
    years: list[int] = [2019, 2020, 2021, 2022, 2023]
    quartrs: list[int] = [1, 2, 3, 4] 

    parser = Parser(years, quartrs)
    print(parser.qrtrs)

if __name__ == "__main__":
    main()