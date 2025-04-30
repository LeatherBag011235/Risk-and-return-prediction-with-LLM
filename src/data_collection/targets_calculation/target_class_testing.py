from src.data_collection.targets_calculation.targets_parser_class import TargetsParser
from src.data_collection.consts import API_KEY, SECRET_KEY
import traceback
import pandas as pd
from datetime import date
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError


tickers = ["SPH", "AAPL", "MSFT", "TSLA", "GOOGL"]
report_dates = ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01", "2024-01-01"]

tickers = ["NSP", "OCC", "IVAC", "FLT", "AWK", "AME", "BRO", "LW", "JKHY", "KEYS", "ROL", "ANET", "CPRT", ]
report_dates = [
    "2018-01-01", "2018-04-01", "2018-07-01", "2018-10-01",
    "2019-01-01", "2019-04-01", "2019-07-01", "2019-10-01",
    "2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01",
    "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01",
    "2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01",
    "2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01",
    "2024-01-01", "2024-04-01", "2024-07-01",

]


client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Define start and end dates
start_date = pd.to_datetime("2017-01-01").date()
end_date = date.today()

# Request hourly bars for S&P 500 index ETF (SPY)
request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
)
try:
    bars = client.get_stock_bars(request)
except APIError as e:
    print("❌ Alpaca API Error")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    print(f"Full content: {str(e)}")  # may show JSON or HTML
# Fetch the data

df = bars.df

# Filter and format
if not df.empty:
    df = df[df.index.get_level_values(0) == "SPY"]
    df.index = df.index.droplevel(0)
    df = df.sort_index()

    # Resample to daily: take the last VWAP per day
    daily_vwap = df['vwap'].resample('1D').last().dropna()
    snp500_daily = pd.DataFrame({"Close": daily_vwap})

    # Ensure tz-awareness and normalize to match TargetsParser expectations
    if snp500_daily.index.tzinfo is None:
        snp500_daily.index = snp500_daily.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        snp500_daily.index = snp500_daily.index.tz_convert("America/New_York")

    snp500_daily.index = snp500_daily.index.normalize()  # normalize to midnight
else:
    snp500_daily = pd.DataFrame()


for ticker in tickers:
    print(f"\n🔍 Testing ticker: {ticker}")
    try:
        parser = TargetsParser(ticker=ticker, report_dates=report_dates, snp_df=snp500_daily, API_KEY=API_KEY, SECRET_KEY=SECRET_KEY)
        print(f"🏷️ Sector: {parser.sector}")
        print(f"Sig alpha: {parser.const_significance}")

        print("✅ Downloaded daily and hourly data.")
        assert not parser.hist_daily.empty, "❌ hist_daily is empty"
        assert not parser.hist_hf.empty, "❌ hist_hourly is empty"
        assert "vwap" in parser.hist_hf.columns, "❌ 'vwap' not in hourly data"

        parser.compute_price_metrics()
        for date_str in report_dates:
            row = parser.assemble_target_row(date_str)
            if row is not None:
                print(f"📈 Abnormal returns for {date_str}: {[row[k] for k in row if 'abn_r' in k]}")
            
            expected_keys = [
                "two_day_r", "three_day_r", "four_day_r", "five_day_r", "six_day_r", "seven_day_r", "full_q_r",
                "two_day_e_r", "three_day_e_r", "four_day_e_r", "five_day_e_r", "six_day_e_r", "seven_day_e_r", "full_q_e_r",
                "two_day_abn_r", "three_day_abn_r", "four_day_abn_r", "five_day_abn_r", "six_day_abn_r", "seven_day_abn_r", "full_q_abn_r",
                "two_day_r_vol", "three_day_r_vol", "four_day_r_vol", "five_day_r_vol", "six_day_r_vol", "seven_day_r_vol", "full_q_r_vol"
            ]


            if row is not None:
                missing = [key for key in expected_keys if key not in row]
                print("❌ Missing keys:", missing)

                has_none = any(v is None for v in row.values())
                print(has_none)

                assert len(row.keys()) == 28, f"❌ Unexpected number of metrics: {len(row)} \n"
                print(f"✅ Metrics for {date_str}: OK")

        parser.compute_eps_surprise()
        parser.compute_firm_size()
        for date_str in report_dates:
            eps = parser.get_eps_and_size(date_str)
            print(f"📊 EPS & Size for {date_str}: {eps}")

        print(f"✅ All tests passed for {ticker}")

    except Exception as e:
        print(f"❌ Test failed for {ticker}: {e}")
        traceback.print_exc()

