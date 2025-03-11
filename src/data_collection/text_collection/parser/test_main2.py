import sys 
sys.path.append(r'C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src')

from data_collection.text_collection.parser.parser_total import ParserTotal

years: list[int] = [2018, 2019, 2020, 2021, 2022, 2023, 2024,]
quartrs: list[int] = [1, 2, 3, 4] 

DB_PARAMS = {
    "dbname": "reports_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

def test_main():
    parser = ParserTotal(years, quartrs, db_params=DB_PARAMS,)
    #parser.get_company_links()
    parser.get_all_lines()            # Fetch all raw SEC lines
    parser.get_all_links()            # Process them into a DataFrame
    parser.filter_type()              # Filters out all filings of the wrong type from self.reports_df
    #parser.filter_existing_entries()  # Filter out all filings that are already in database
    parser.get_cik_ticker_mapping()   # Matches cik numbers and ticker from a separate SEC page
    parser.store_reports_in_db()  

if __name__ == "__main__":
    test_main()