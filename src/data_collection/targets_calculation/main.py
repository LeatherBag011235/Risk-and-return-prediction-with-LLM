from src.data_collection.targets_calculation.targets_class import YFinParser

ticker = "AAPL"
report_dates = ["2024-04-01", "2024-08-01", "2024-12-15"]

def main():
    target = YFinParser(ticker=ticker, report_dates=report_dates)

    print(target.download_hourly("2024-05-01", "2025-01-01"))

if __name__ == "__main__":
    main()
