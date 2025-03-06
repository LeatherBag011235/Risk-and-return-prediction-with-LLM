import sys 
sys.path.append(r'C:\Users\Maxim Shibanov\Projects_Py\Risk-and-return-prediction-with-LLM\src')

from data_collection.parser.parser_total import ParserTotal

years: list[int] = [2023, ]
quartrs: list[int] = [3, 4,] 

DB_PARAMS = {
    "dbname": "reports_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

def test_main():
    parser = ParserTotal(years, quartrs, db_params=DB_PARAMS,)
    parser.get_all_lines()           # Fetch all raw SEC lines
    parser.get_all_links()         # Process them into a DataFrame
    parser.filter_type()             # Filters out all filings of the wrong type from self.reports_df
    parser.check_remaining_companies()
    parser.filter_existing_entries() # Filter out all filings that are already in database
    parser.store_reports_in_db()

if __name__ == "__main__":
    test_main()