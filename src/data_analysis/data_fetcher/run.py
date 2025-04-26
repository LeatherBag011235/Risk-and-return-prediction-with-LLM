from src.data_analysis.data_fetcher.data_fetcher_class import DataFetcher
from src.data_collection.consts import  DB_PARAMS


def main(): 
    fetcher = DataFetcher(DB_PARAMS)

    df = fetcher.fetch_data(
    regressors=['lm_orig_score', 'hv_orig_score', 'eps_surprise', 'f_size'],
    company_filters={'sector': 'Technology'},
    prepare_fixed_effects=True
)

    print(df.head())

if __name__ == "__main__":
    main()
