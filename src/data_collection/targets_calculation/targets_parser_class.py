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


class TargetsParser:
    def __init__(self, ticker, report_dates, snp_hourly_raw, API_KEY, SECRET_KEY):
        self.ticker = ticker
        self.report_dates = sorted(report_dates)
        self.api_key = API_KEY
        self.secret_key = SECRET_KEY

        self.company = yf.Ticker(ticker)
        self.sector = self.company.info.get("sector", None)

        # --- Retrieve and store hourly data ---
        self.hist_hf = self._download_hourly_data()
        self.snp_hf = snp_hourly_raw.copy()

        # --- Apply identical transformation to both datasets ---
        self.hist_daily = self._resample_and_format(self.hist_hf)
        self.snp500 = self._resample_and_format(self.snp_hf)

        # --- Sanity check ---
        self._ensure_index_alignment()

        self.ff_factors = self._load_fama_french_factors()
        self._estimate_factor_model()

        self.end_dates = {}
        self.returns = {}
        self.eps_surprises = {}
        self.firm_sizes = {}
    
    def _download_hourly_data(self):
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

    def _resample_and_format(self, df):
        df = df.copy()
        df = df.sort_index()

        daily_vwap = df['vwap'].resample('1D').last().dropna()

        if daily_vwap.index.tz is None:
            daily_vwap.index = daily_vwap.index.tz_localize("UTC")

        daily_vwap.index = daily_vwap.index.tz_convert("America/New_York").normalize()

        daily_df = pd.DataFrame({"Close": daily_vwap})
        daily_df["Return"] = daily_df["Close"].pct_change() * 100
        return daily_df
    
    def _ensure_index_alignment(self):
        hist_index = self.hist_daily.index
        snp_index = self.snp500.index

        # Find any dates in hist that are not in snp
        missing_dates = hist_index.difference(snp_index)

        if not missing_dates.empty:
            logging.error(f"❌ {len(missing_dates)} dates in {self.ticker} data missing from S&P 500 index!")
            logging.debug(f"Missing dates: {missing_dates.tolist()}")
            raise ValueError(f"❌ Misaligned indexes between {self.ticker} and S&P 500 — fix needed.")

        logging.debug(f"✅ All dates in {self.ticker} data are present in S&P 500 index.")

    def _load_fama_french_factors(self):
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
        """
        combined = self.hist_daily.join(self.ff_factors, how="inner").dropna()
        y = combined["Return"] - combined["RF"]
        X = combined[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        self.factor_model = model

        p_val = model.pvalues.get("const", np.nan)
        const_value = model.params.get("const", np.nan)

        self.const_significance = {
            "value": const_value,
            "0.05": bool(p_val < 0.05),
            "0.01": bool(p_val < 0.01),
            "0.001": bool(p_val < 0.001)
        }

    def _get_nearest_trading_day(self, date_val: str | pd.Timestamp) -> pd.Timestamp | None:
        """
        Find the next available trading day.

        Args:
            date_val: Target date.

        Returns:
            Nearest trading day or None.
        """
        date_val = pd.to_datetime(date_val)
        max_date = self.hist_daily.index.max()

        if date_val.tzinfo is None:
            date_val = date_val.tz_localize("America/New_York")
        else:
            date_val = date_val.tz_convert("America/New_York")

        while date_val not in self.hist_daily.index:
            date_val += pd.Timedelta(days=1)
            if date_val > max_date:
                print(f"No data for {self.ticker} from {date_val}")
                return None
        return date_val

    def _find_end_price(self, start_index: int) -> tuple[list[float | None], list[pd.Timestamp | None]]:
        """
        Find stock prices 2-7 days after a starting point.

        Args:
            start_index: Start index in daily data.

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
    
    def _find_benchmark_prices(self, start_date, end_dates):
        try:
            idx = self.snp500.index.get_loc(start_date)
        except KeyError:
            print(f"❌ start_date {start_date} not found in S&P 500 index!")
            raise

        start_idx = self.snp500.index.get_loc(start_date)
        snp_start_price = self.snp500.iloc[start_idx]['Close']
        snp_end_price_list = []

        for end_date in end_dates:
            if end_date is not None:
                try:
                    end_idx = self.snp500.index.get_loc(end_date)
                except KeyError:
                    print(f"❌ end_date {end_date} not found in S&P 500 index!")
                    raise
                end_idx = self.snp500.index.get_loc(end_date)
                snp_end_price_list.append(self.snp500.iloc[end_idx]['Close'])
            else:
                snp_end_price_list.append(None)

        return snp_start_price, snp_end_price_list

    def _find_quarter_end(self, date_str: str, start_date: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None, int | None]:
        """
        Find end of the financial quarter.

        Args:
            date_str: Report date.
            start_date: Start trading day.

        Returns:
            Tuple of (end price, end date, number of trading days).
        """
        idx = self.report_dates.index(date_str)

        if idx == len(self.report_dates) - 1:
            return None, None, None

        next_date = self.report_dates[idx + 1]
        next_date = self._get_nearest_trading_day(next_date)

        if next_date is None:
            return None, None, None

        end_index = self.hist_daily.index.get_loc(next_date)
        start_index = self.hist_daily.index.get_loc(start_date)

        return self.hist_daily.iloc[end_index]['Close'], next_date, end_index - start_index

    def _calc_pct_returns(self, start_price: float, end_prices: list[float | None]) -> list[float | None]:
        return [((p - start_price) / start_price) * 100 if p is not None else None for p in end_prices]

    def _calc_volatility(self, start_date: pd.Timestamp, end_dates: list[pd.Timestamp | None]) -> list[float | None]:
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

                X = ff_window[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]]
                X = sm.add_constant(X, has_constant='add')
                pred = self.factor_model.predict(X)

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
        Compute end dates for short-term and quarterly periods for each report date.
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
        Compute normalized returns, excess returns, abnormal returns, and realized volatility.
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

            excess_returns = [a - b if a is not None and b is not None else None
                              for a, b in zip(reg_returns, snp_returns)]

            timeframe_lengths = [2, 3, 4, 5, 6, 7, q_len]

            normalized_returns = [x / y if x is not None and y is not None else None
                                  for x, y in zip(reg_returns, timeframe_lengths)]

            normalized_excess_returns = [x / y if x is not None and y is not None else None
                                         for x, y in zip(excess_returns, timeframe_lengths)]

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
        Download and map EPS surprises to report dates.
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
        Download and map firm size (market capitalization) to report dates.
        """
        try:
            shares_df = self.company.get_shares_full(start=self.report_dates[0], end=None)
            for rep_date in self.report_dates:
                ts = self._get_nearest_trading_day(rep_date)
                price = self.hist_daily.loc[ts]['Close']
                closest = min(shares_df.index, key=lambda x: abs(x - ts))
                shares = shares_df.loc[closest]
                if isinstance(shares, (int, np.integer)):
                    self.firm_sizes[rep_date] = shares * price
        except Exception as e:
            logging.error(f"Firm size calc failed for {self.ticker}: {e}")

    def assemble_target_row(self, date_str: str) -> dict[str, float | None] | None:
        """
        Assemble dictionary of computed metrics for a given report date.

        Args:
            date_str: Report date string.

        Returns:
            Dictionary of metrics or None if missing data.
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