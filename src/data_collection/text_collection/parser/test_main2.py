from src.data_collection.text_collection.parser.parser_total import ParserTotal

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
    parser.get_company_links()

if __name__ == "__main__":
    test_main()