import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
import statsmodels.api as sm
from datetime import date
from pathlib import Path
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.data_collection.targets_calculation.logging_config import logger


class TargetsParser:
    """
    Parses and computes return-based financial metrics for a given stock around earnings report dates.

    It performs the following:
    - Downloads and resamples high-frequency stock and market data.
    - Aligns stock data with S&P 500 benchmark data.
    - Loads and applies the Fama-French 5-factor model.
    - Computes percentage returns, excess returns, abnormal returns, and realized volatility.
    - Extracts earnings surprises and firm size data.
    - Constructs structured output rows for insertion into a database.
    """
    def __init__(self, ticker: str, report_dates: list[str], snp_hourly_raw: pd.DataFrame, api_key: str, secret_key: str):
        self.ticker = ticker
        self.report_dates = sorted(report_dates)
        logger.debug(f"Sorted repport dates: {self.report_dates}")
        self.api_key = api_key
        self.secret_key = secret_key

        self.company = yf.Ticker(ticker)
        self.sector = self.company.info.get("sector", None)

        self.hist_hf = self._download_hourly_data()
        self.snp_hf = snp_hourly_raw.copy()

        self.hist_daily = self._resample_and_format(self.hist_hf)
        self.snp500 = self._resample_and_format(self.snp_hf)

        self._ensure_index_alignment()

        self.ff_factors = self._load_fama_french_factors()
        self._estimate_factor_model()

        self.end_dates: dict[str, dict] = {}
        self.returns: dict[str, dict | None] = {}
        self.eps_surprises: dict[str, float | None] = {}
        self.firm_sizes: dict[str, float | None] = {}

    def _download_hourly_data(self) -> pd.DataFrame:
        """
        Download hourly stock data from Alpaca.

        Returns:
            DataFrame with hourly bars.
        """
        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        start_date = pd.to_datetime(min(self.report_dates)).date()
        end_date = date.today()

        request = StockBarsRequest(
            symbol_or_symbols=[self.ticker],
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
        )

        bars = client.get_stock_bars(request).df
        assert not bars.empty, f"❌ No hourly data returned for {self.ticker}"

        bars = bars[bars.index.get_level_values(0) == self.ticker]
        bars.index = bars.index.droplevel(0)
        bars = bars.sort_index()
        return bars

    def _resample_and_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample hourly data to daily VWAP.

        Args:
            df: Hourly bar data.

        Returns:
            Daily VWAP DataFrame with returns.
        """
        df = df.copy()
        df = df.sort_index()

        daily_vwap = df['vwap'].resample('1D').last().dropna()

        if daily_vwap.index.tz is None:
            daily_vwap.index = daily_vwap.index.tz_localize("UTC")

        daily_vwap.index = daily_vwap.index.tz_convert("America/New_York").normalize()

        daily_df = pd.DataFrame({"Close": daily_vwap})
        daily_df["Return"] = daily_df["Close"].pct_change() * 100
        return daily_df

    def _ensure_index_alignment(self) -> None:
        """
        Ensure that historical and benchmark indexes align on dates.
        Raises:
            ValueError: If alignment fails.
        """
        hist_index = self.hist_daily.index
        snp_index = self.snp500.index

        missing_dates = hist_index.difference(snp_index)

        if not missing_dates.empty:
            logging.error(f"\u274c {len(missing_dates)} dates in {self.ticker} data missing from S&P 500 index!")
            logging.debug(f"Missing dates: {missing_dates.tolist()}")
            raise ValueError(f"\u274c Misaligned indexes between {self.ticker} and S&P 500 — fix needed.")

        logging.debug(f"\u2705 All dates in {self.ticker} data are present in S&P 500 index.")

    def _load_fama_french_factors(self) -> pd.DataFrame:
        """
        Load Fama-French 5-factor data from local CSV.

        Returns:
            Parsed DataFrame with date index.
        """
        base_path = Path(__file__).parent
        file_path = base_path / 'F-F_Research_Data_5_Factors_2x3_daily.CSV'

        df = pd.read_csv(file_path)
        df.columns = [col.strip() for col in df.columns]
        df.rename(columns={"Mkt-RF": "Mkt_RF"}, inplace=True)

        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index = df.index.tz_localize("America/New_York").normalize()

        return df

    def _estimate_factor_model(self) -> None:
        """
        Estimate Fama-French 5-factor model with intercept.
        Computes model coefficients and significance levels.
        """
        combined = self.hist_daily.join(self.ff_factors, how="inner").dropna()
        y = combined["Return"] - combined["RF"]
        x = combined[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]]
        x = sm.add_constant(x)

        model = sm.OLS(y, x).fit()
        self.factor_model = model

        p_val = model.pvalues.get("const", np.nan)
        const_value = model.params.get("const", np.nan)

        self.const_significance = {
            "value": const_value,
            "0.05": bool(p_val < 0.05),
            "0.01": bool(p_val < 0.01),
            "0.001": bool(p_val < 0.001)
        }

    
    def _get_nearest_trading_day(self, date: str | pd.Timestamp, direction: str ="forward") -> pd.Timestamp | None:
        """
        Return the nearest trading day on or after date_val if within data bounds.

        Args:
            date_val: The reference date to align to a valid trading day.

        Returns:
            Nearest valid trading day or None if out of bounds.
        """
        date = pd.to_datetime(date)
        max_date = self.hist_daily.index.max()
        min_date = self.hist_daily.index.min()

        if date.tzinfo is None:
            date = date.tz_localize("America/New_York")
        else:
            date = date.tz_convert("America/New_York")

        date = date.normalize()

        if not self.hist_daily.index.is_monotonic_increasing:
            raise RuntimeError("❌ hist_daily index is not sorted!")

        if date < min_date or date > max_date:
            logger.warning(f"{self.ticker}: date {date} is out of bounds ({min_date} to {max_date})")
            return None

        hist_idx = self.hist_daily.index

        if direction == "forward":
            idx = hist_idx.get_indexer([date], method="backfill")[0]
        elif direction == "backward":
            idx = hist_idx.get_indexer([date], method="pad")[0]
        else:
            raise ValueError(f"Invalid direction '{direction}', expected 'forward' or 'backward'.")

        if idx == -1:
            return None

        return hist_idx[idx]


    def _find_end_price(self, start_index: int) -> tuple[list[float | None], list[pd.Timestamp | None]]:
        """
        Find prices and dates 2 to 7 days after a given index.

        Args:
            start_index: Index in historical data.

        Returns:
            Tuple of list of prices and list of dates.
        """
        prices, dates = [], []
        for x in range(2, 8):
            idx = start_index + x
            if idx < len(self.hist_daily):
                prices.append(self.hist_daily.iloc[idx]['Close'])
                dates.append(self.hist_daily.index[idx])
            else:
                prices.append(None)
                dates.append(None)
        return prices, dates

    def _find_benchmark_prices(self, start_date: pd.Timestamp, end_dates: list[pd.Timestamp | None]) -> tuple[float, list[float | None]]:
        """
        Fetch S&P 500 start and end prices matching the analysis window.

        Args:
            start_date: Start date for benchmark.
            end_dates: Corresponding end dates.

        Returns:
            Tuple of start price and list of end prices.
        """
        try:
            start_idx = self.snp500.index.get_loc(start_date)
        except KeyError:
            print(f"❌ start_date {start_date} not found in S&P 500 index!")
            raise

        snp_start_price = self.snp500.iloc[start_idx]['Close']
        snp_end_price_list = []

        for end_date in end_dates:
            if end_date is not None:
                try:
                    end_idx = self.snp500.index.get_loc(end_date)
                    snp_end_price_list.append(self.snp500.iloc[end_idx]['Close'])
                except KeyError:
                    print(f"❌ end_date {end_date} not found in S&P 500 index!")
                    snp_end_price_list.append(None)
            else:
                snp_end_price_list.append(None)

        return snp_start_price, snp_end_price_list

    def _find_quarter_end(self, date_str: str, start_date: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None, int | None]:
        """
        Determine the end of the financial quarter following a report.

        Args:
            date_str: Current report date.
            start_date: Start of measurement period.

        Returns:
            Final price, date, and trading days in quarter.
        """
        idx = self.report_dates.index(date_str)

        if idx == len(self.report_dates) - 1:
            return None, None, None

        next_date = self.report_dates[idx + 1]
        logger.debug(f"date_str: {date_str}, next date_str: {next_date}, ")

        next_date = self._get_nearest_trading_day(next_date, direction="backward")

        if next_date is None:
            return None, None, None
        
        if next_date <= start_date:
            logger.warning(f"❌ Invalid quarter for {self.ticker}: start={start_date}, end={next_date}")
            return None, None, None
        
        logger.debug(f"start_date: {start_date}next_date: {next_date}")

        end_index = self.hist_daily.index.get_loc(next_date)
        start_index = self.hist_daily.index.get_loc(start_date)

        q_len = end_index - start_index

        if q_len < 1:
            logger.warning(f"q_len for start: {start_date} for rep_date-{date_str} -- end: {next_date} is {q_len}")
            raise ValueError(f"q_len for {self.ticker} is {q_len}")

        return self.hist_daily.iloc[end_index]['Close'], next_date, q_len

    def _calc_pct_returns(self, start_price: float, end_prices: list[float | None]) -> list[float | None]:
        """
        Calculate percentage returns.

        Args:
            start_price: Initial price.
            end_prices: List of ending prices.

        Returns:
            List of returns.
        """
        return [((p - start_price) / start_price) * 100 if p is not None else None for p in end_prices]

    def _calc_volatility(self, start_date: pd.Timestamp, end_dates: list[pd.Timestamp | None]) -> list[float | None]:
        """
        Calculate realized volatility using hourly VWAP data.

        Args:
            start_date: Start date.
            end_dates: List of end dates.

        Returns:
            List of volatilities.
        """
        df = self.hist_hf.copy()
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('1h')
        df['vwap'] = df['vwap'].ffill()
        df['log_return'] = np.log(df['vwap'] / df['vwap'].shift(1))
        df = df.dropna(subset=['log_return'])

        vols = []
        for end_date in end_dates:
            if end_date:
                window = df.loc[start_date:end_date]
                if window.shape[0] < 2:
                    vols.append(None)
                    continue
                realized_vol = window['log_return'].std(ddof=0) * np.sqrt(252 * 6.5)
                vols.append(realized_vol)
            else:
                vols.append(None)
        return vols

    def _compute_abnormal_returns(self, start_date: pd.Timestamp, end_date_list: list[pd.Timestamp | None]) -> list[float | None]:
        """
        Compute abnormal returns over given windows using Fama-French predicted values.

        Args:
            start_date: Beginning of window.
            end_date_list: List of end dates.

        Returns:
            List of abnormal returns.
        """
        ab_norm_ret = []

        for end in end_date_list:
            if end is not None:
                window = pd.date_range(start_date, end, freq='B')
                if window.tz is None:
                    window = window.tz_localize("America/New_York")
                window = window.tz_convert("America/New_York").normalize()

                ff_window = self.ff_factors.loc[self.ff_factors.index.isin(window)]

                if ff_window.empty:
                    ab_norm_ret.append(None)
                    continue

                x = ff_window[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]]
                x = sm.add_constant(x, has_constant='add')
                pred = self.factor_model.predict(x)

                expected_return = pred.sum() + ff_window["RF"].sum()
                actual_window = self.hist_daily.loc[self.hist_daily.index.isin(window)]

                if actual_window.empty:
                    ab_norm_ret.append(None)
                    continue

                actual_return = actual_window["Return"].sum()
                ab_ret = actual_return - expected_return
                ab_norm_ret.append(ab_ret / len(ff_window))
            else:
                ab_norm_ret.append(None)

        return ab_norm_ret

    def compute_end_dates(self) -> None:
        """
        Compute end dates and prices for short-term (2-7 days) and quarterly windows for each report date.
        Populates the `end_dates` dictionary with start/end price info and lengths.
        """
        for date_str in self.report_dates:
            start_date = self._get_nearest_trading_day(date_str)

            if start_date is not None:
                start_index = self.hist_daily.index.get_loc(start_date)

                end_price_list, end_date_list = self._find_end_price(start_index)
                q_end_price, q_end_date, q_len = self._find_quarter_end(date_str, start_date)
                end_price_list.append(q_end_price)
                end_date_list.append(q_end_date)

                self.end_dates[date_str] = {
                    "start_date": start_date,
                    "start_index": start_index,
                    "end_prices": end_price_list,
                    "end_dates": end_date_list,
                    "q_len": q_len
                }
            else:
                self.end_dates[date_str] = None

    def compute_price_metrics(self) -> None:
        """
        Compute normalized returns, excess returns, abnormal returns, and realized volatilities.
        Populates the `returns` dictionary for each report date.
        """
        self.compute_end_dates()

        for date_str, info in self.end_dates.items():
            if info is None:
                self.returns[date_str] = None
                continue

            start_date = info["start_date"]
            start_index = info["start_index"]
            start_price = self.hist_daily.iloc[start_index]['Close']

            end_price_list = info["end_prices"]
            end_date_list = info["end_dates"]
            q_len = info["q_len"]

            snp_start_price, snp_end_price_list = self._find_benchmark_prices(start_date, end_date_list)

            reg_returns = self._calc_pct_returns(start_price, end_price_list)
            snp_returns = self._calc_pct_returns(snp_start_price, snp_end_price_list)

            excess_returns = [a - b if a is not None and b is not None else None for a, b in zip(reg_returns, snp_returns)]

            timeframe_lengths = [2, 3, 4, 5, 6, 7, q_len]
            normalized_returns = [x / y if x is not None and y is not None else None for x, y in zip(reg_returns, timeframe_lengths)]
            normalized_excess_returns = [x / y if x is not None and y is not None else None for x, y in zip(excess_returns, timeframe_lengths)]

            vol = self._calc_volatility(start_date, end_date_list)
            norm_abnormal_returns = self._compute_abnormal_returns(start_date, end_date_list)

            self.returns[date_str] = {
                "reg": normalized_returns,
                "excess": normalized_excess_returns,
                "vol": vol,
                "abn": norm_abnormal_returns,
                "q_len": q_len
            }

    def compute_eps_surprise(self) -> None:
        """
        Download and assign EPS surprises to report dates.
        Populates the `eps_surprises` dictionary.
        """
        try:
            eps = self.company.get_earnings_dates(limit=10000).reset_index().drop_duplicates()
            for rep_date in self.report_dates:
                rep_ts = pd.Timestamp(rep_date, tz='America/New_York')
                closest = min(eps['Earnings Date'], key=lambda x: abs(x - rep_ts))
                self.eps_surprises[rep_date] = eps.loc[eps['Earnings Date'] == closest, 'Surprise(%)'].values[0]
        except Exception as e:
            logging.error(f"EPS surprise fetch failed for {self.ticker}: {e}")

    def compute_firm_size(self) -> None:
        """
        Download and assign market capitalization to report dates.
        Populates the `firm_sizes` dictionary.
        """
        try:
            shares_df = self.company.get_shares_full(start=self.report_dates[0], end=None)
            for rep_date in self.report_dates:
                ts = self._get_nearest_trading_day(rep_date)
                if ts is None:
                    self.firm_sizes[rep_date] = None
                else:
                    price = self.hist_daily.loc[ts]['Close']
                    closest = min(shares_df.index, key=lambda x: abs(x - ts))
                    shares = shares_df.loc[closest]
                    if isinstance(shares, (int, np.integer)):
                        self.firm_sizes[rep_date] = shares * price
        except Exception as e:
            logging.error(f"Firm size calc failed for {self.ticker}: {e}")

    def assemble_target_row(self, date_str: str) -> dict[str, float | None] | None:
        """
        Assemble a metrics dictionary for the given report date.

        Args:
            date_str: Report date string.

        Returns:
            Dictionary with computed metrics, or None.
        """
        r = self.returns.get(date_str)
        if r is None:
            logging.info(f"⚠️ Skipping {date_str}: return data is None")
            return None

        return {
            "two_day_r": r["reg"][0],
            "three_day_r": r["reg"][1],
            "four_day_r": r["reg"][2],
            "five_day_r": r["reg"][3],
            "six_day_r": r["reg"][4],
            "seven_day_r": r["reg"][5],
            "full_q_r": r["reg"][6],

            "two_day_e_r": r["excess"][0],
            "three_day_e_r": r["excess"][1],
            "four_day_e_r": r["excess"][2],
            "five_day_e_r": r["excess"][3],
            "six_day_e_r": r["excess"][4],
            "seven_day_e_r": r["excess"][5],
            "full_q_e_r": r["excess"][6],

            "two_day_abn_r": r["abn"][0],
            "three_day_abn_r": r["abn"][1],
            "four_day_abn_r": r["abn"][2],
            "five_day_abn_r": r["abn"][3],
            "six_day_abn_r": r["abn"][4],
            "seven_day_abn_r": r["abn"][5],
            "full_q_abn_r": r["abn"][6],

            "two_day_r_vol": r["vol"][0],
            "three_day_r_vol": r["vol"][1],
            "four_day_r_vol": r["vol"][2],
            "five_day_r_vol": r["vol"][3],
            "six_day_r_vol": r["vol"][4],
            "seven_day_r_vol": r["vol"][5],
            "full_q_r_vol": r["vol"][6],
        }

    def get_eps_and_size(self, date_str: str) -> dict[str, float | None]:
        """
        Get EPS surprise and firm size for a report date.

        Args:
            date_str: Report date string.

        Returns:
            Dictionary with 'eps_surprise' and 'f_size'.
        """
        return {
            "eps_surprise": self.eps_surprises.get(date_str),
            "f_size": self.firm_sizes.get(date_str)
        }