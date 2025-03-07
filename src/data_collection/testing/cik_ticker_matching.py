import requests
import pandas as pd

# SEC JSON URL
SEC_CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

def get_cik_ticker_mapping():
    response = requests.get(SEC_CIK_TICKER_URL, headers={"User-Agent": "my-email@example.com"})
    data = response.json()

    # Convert JSON into a Pandas DataFrame
    cik_ticker_df = pd.DataFrame.from_dict(data, orient="index")

    # Normalize CIK (leading zeros)
    cik_ticker_df['cik'] = cik_ticker_df['cik_str'].apply(lambda x: str(x).zfill(10))

    return cik_ticker_df[['cik', 'ticker']]

# Fetch mapping
cik_ticker_df = get_cik_ticker_mapping()
print(cik_ticker_df.head())
