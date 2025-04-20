import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import date
import logging
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.data_collection.consts import API_KEY, SECRET_KEY

class TargetsParser:
    def __init__(self, ticker, report_dates, snp_df):
        self.ticker = ticker
        self.report_dates = sorted(report_dates)

        # --- Set up S&P 500 daily dataframe ---
        self.snp500 = snp_df.copy()

        assert not self.snp500.empty, f"❌ S&P 500 daily data is empty"
        assert isinstance(self.snp500.index, pd.DatetimeIndex), "❌ S&P 500 index is not a DatetimeIndex"

        if self.snp500.index.tz is None:
            self.snp500.index = self.snp500.index.tz_localize("UTC")

        self.snp500.index = self.snp500.index.tz_convert("America/New_York").normalize()

        # --- Set up ticker hourly data ---
        self.company = yf.Ticker(ticker)
        client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

        start_date = pd.to_datetime(min(self.report_dates)).date()
        end_date = date.today()

        request = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        assert not df.empty, f"❌ No hourly data returned for {ticker}"

        df = df[df.index.get_level_values(0) == ticker]
        df.index = df.index.droplevel(0)
        df = df.sort_index()

        self.hist_hf = df.copy()

        # --- Build daily VWAP from hourly bars ---
        daily_vwap = df['vwap'].resample('1D').last().dropna()
        if daily_vwap.index.tz is None:
            daily_vwap.index = daily_vwap.index.tz_localize("UTC")
        daily_vwap.index = daily_vwap.index.tz_convert("America/New_York").normalize()

        self.hist_daily = pd.DataFrame({"Close": daily_vwap})

        # --- Final alignment check ---
        assert not self.hist_daily.empty, f"❌ Daily VWAP is empty for {ticker}"
        assert self.hist_daily.index.intersection(self.snp500.index).size > 0, \
            f"❌ No overlapping dates between {ticker} and S&P 500"

        logging.debug(f"✅ Earliest stock date:", self.hist_daily.index.min())
        logging.debug(f"✅ Earliest S&P 500 date:", self.snp500.index.min())
        logging.debug(f"✅ Index overlap:", self.hist_daily.index[0] in self.snp500.index)

        self.returns = {}
        self.eps_surprises = {}
        self.firm_sizes = {}


    def _get_nearest_trading_day(self, date):
        date = pd.to_datetime(date)
        max_date = self.hist_daily.index.max()

        if date.tzinfo is None:
            date = date.tz_localize("America/New_York")
        else:
            date = date.tz_convert("America/New_York")

        while date not in self.hist_daily.index:
            date += pd.Timedelta(days=1)
            if date > max_date:
                raise ValueError(f"⚠️ Could not find trading day for {date} (past max available: {max_date})")


        return date

    
    def _find_end_price(self, start_index):
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
        start_idx = self.snp500.index.get_loc(start_date)
        snp_start_price = self.snp500.iloc[start_idx]['Close']
        snp_end_price_list = []

        for end_date in end_dates:
            if end_date is not None:
                end_idx = self.snp500.index.get_loc(end_date)
                end_price = self.snp500.iloc[end_idx]['Close']
                snp_end_price_list.append(end_price)
            else:
                snp_end_price_list.append(None)
                
            
        return snp_start_price, snp_end_price_list
    
    def _find_quarter_end(self, date_str, start_date):
        idx = self.report_dates.index(date_str)

        # Last report date — no next quarter
        if idx == len(self.report_dates) - 1:
            return None, None, None

        next_date = self.report_dates[idx + 1]
        next_date = self._get_nearest_trading_day(next_date)

        assert next_date in self.hist_daily.index, f"Next report date '{next_date}' not found in daily data."

        end_index = self.hist_daily.index.get_loc(next_date)
        start_index = self.hist_daily.index.get_loc(start_date)

        return self.hist_daily.iloc[end_index]['Close'], next_date, end_index - start_index

        
    def _calc_log_returns(self, start_price, end_prices):
        return [np.log(p / start_price) if p else None for p in end_prices]

    def _calc_volatility(self, start_date, end_dates):
        df = self.hist_hf.copy()

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = df.asfreq('1h') 
        df['vwap'] = df['vwap'].ffill()

        df['log_return'] = np.log(df['vwap'] / df['vwap'].shift(1))
        df = df.dropna(subset=['log_return'])

        vols = []

        for end_date in end_dates:
            if end_date:
                end_date = pd.to_datetime(end_date)

                window = df.loc[start_date:end_date]

                if window.shape[0] < 2:
                    print(f"⚠️ Not enough data from {start_date} to {end_date}")
                    vols.append(None)
                    continue

                realized_vol = window['log_return'].std(ddof=0) * np.sqrt(252 * 6.5)
                vols.append(realized_vol)
            else:
                vols.append(None)

        return vols

    def compute_price_metrics(self):
        for date_str in self.report_dates:
            start_date = self._get_nearest_trading_day(date_str)
            start_index = self.hist_daily.index.get_loc(start_date)
            start_price = self.hist_daily.iloc[start_index]['Close']

            end_price_list, end_date_list = self._find_end_price(start_index)

            q_end_price, q_end_date, q_len = self._find_quarter_end(date_str, start_date)
            end_price_list.append(q_end_price)
            end_date_list.append(q_end_date)

            snp_start_price, snp_end_price_list = self._find_benchmark_prices(start_date, end_date_list)

            reg_returns = self._calc_log_returns(start_price, end_price_list)
            snp_returns = self._calc_log_returns(snp_start_price, snp_end_price_list)

            excess_returns = [a - b if a is not None and b is not None else None
                              for a, b in zip(reg_returns, snp_returns)]
            
            timeframe_lengths = [2, 3, 4, 5, 6, 7, q_len]
             
            normalized_returns = [
                x / y if x is not None and y is not None else None
                for x, y in zip(reg_returns, timeframe_lengths)
            ]
            normalized_excess_returns = [
                x / y if x is not None and y is not None else None
                for x, y in zip(excess_returns, timeframe_lengths)
            ]
            
            vol = self._calc_volatility(start_date, end_date_list)

            self.returns[date_str] = {
                "reg": normalized_returns,
                "excess": normalized_excess_returns,
                "vol": vol,
                "q_len": q_len
            }

    def compute_eps_surprise(self):
        try:
            eps = self.company.get_earnings_dates(limit=10000).reset_index().drop_duplicates()
            for rep_date in self.report_dates:
                rep_ts = pd.Timestamp(rep_date, tz='America/New_York')
                closest = min(eps['Earnings Date'], key=lambda x: abs(x - rep_ts))
                self.eps_surprises[rep_date] = eps.loc[eps['Earnings Date'] == closest, 'Surprise(%)'].values[0]
        except Exception as e:
            logging.error(f"EPS surprise fetch failed for {self.ticker}: {e}")

    def compute_firm_size(self):
        try:
            shares_df = self.company.get_shares_full(start=self.report_dates[0], end=None)
            for rep_date in self.report_dates:
                ts = self._get_nearest_trading_day(rep_date)
                price = self.hist_daily.loc[ts]['Close']
                closest = min(shares_df.index, key=lambda x: abs(x - ts))
                shares = shares_df.loc[closest]
                if isinstance(shares, np.int64):
                    self.firm_sizes[rep_date] = shares * price
        except Exception as e:
            logging.error(f"Firm size calc failed for {self.ticker}: {e}")

    def assemble_target_row(self, date_str):
        r = self.returns[date_str]
        row = {
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
            "two_day_r_vol": r["vol"][0],
            "three_day_r_vol": r["vol"][1],
            "four_day_r_vol": r["vol"][2],
            "five_day_r_vol": r["vol"][3],
            "six_day_r_vol": r["vol"][4],
            "seven_day_r_vol": r["vol"][5],
            "full_q_r_vol": r["vol"][6],
        }
        return row

    def get_eps_and_size(self, date_str):
        return {
            "eps_surprise": self.eps_surprises.get(date_str),
            "f_size": self.firm_sizes.get(date_str)
        }
