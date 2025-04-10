import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging

class CompanyReportAnalyzer:
    def __init__(self, ticker, report_dates):
        self.ticker = ticker
        self.report_dates = sorted(report_dates)
        self.company = yf.Ticker(ticker)
        self.hist = self.company.history(period="max")
        self.snp500 = yf.Ticker("^GSPC").history(period="max", auto_adjust=True)
        self.returns = {}
        self.eps_surprises = {}
        self.firm_sizes = {}

    def _get_nearest_trading_day(self, date):
        date = pd.Timestamp(date, tz='America/New_York')
        while date not in self.hist.index:
            date += pd.Timedelta(days=1)
        return date

    def compute_price_metrics(self):
        for date_str in self.report_dates:
            start_date = self._get_nearest_trading_day(date_str)
            start_index = self.hist.index.get_loc(start_date)
            start_price = self.hist.iloc[start_index]['Open']

            # Collect end prices for days 2-7
            end_price_list, end_date_list = self._find_end_price(start_index)

            # Full quarter return
            q_end_price, q_end_date, q_len = self._find_quarter_end(date_str, start_date)
            end_price_list.append(q_end_price)
            end_date_list.append(q_end_date)

            reg_returns = self._calc_log_returns(start_price, end_price_list)
            snp_returns = self._calc_benchmark_returns(start_date, end_date_list)
            excess_returns = [a - b if a is not None and b is not None else None
                              for a, b in zip(reg_returns, snp_returns)]
            vol = self._calc_volatility(start_date, end_date_list)

            self.returns[date_str] = {
                "reg": reg_returns,
                "excess": excess_returns,
                "vol": vol,
                "q_len": q_len
            }

    def _find_end_price(self, start_index):
        prices, dates = [], []
        for x in range(2, 8):
            idx = start_index + x
            if idx < len(self.hist):
                prices.append(self.hist.iloc[idx]['Open'])
                dates.append(self.hist.index[idx])
            else:
                prices.append(None)
                dates.append(None)
        return prices, dates

    def _find_quarter_end(self, date_str, start_date):
        try:
            idx = self.report_dates.index(date_str)
            next_date = self.report_dates[idx + 1]
            next_date = self._get_nearest_trading_day(next_date)
            end_index = self.hist.index.get_loc(next_date)
            return self.hist.iloc[end_index]['Open'], next_date, end_index - self.hist.index.get_loc(start_date)
        except IndexError:
            return None, None, None

    def _calc_log_returns(self, start_price, end_prices):
        return [np.log(p / start_price) if p else None for p in end_prices]

    def _calc_benchmark_returns(self, start_date, end_dates):
        start_idx = self.snp500.index.get_loc(start_date)
        start_price = self.snp500.iloc[start_idx]['Open']
        returns = []
        for end_date in end_dates:
            if end_date is None:
                returns.append(None)
                continue
            end_idx = self.snp500.index.get_loc(end_date)
            end_price = self.snp500.iloc[end_idx]['Open']
            returns.append(np.log(end_price / start_price))
        return returns

    def _calc_volatility(self, start_date, end_dates):
        vols = []
        for end_date in end_dates:
            if end_date is None:
                vols.append(None)
                continue
            window = self.hist.loc[start_date:end_date]['Open']
            if len(window) > 1:
                log_returns = np.log(window / window.shift(1)).dropna()
                vols.append(log_returns.std())
            else:
                vols.append(None)
        return vols

    def compute_eps_surprise(self):
        try:
            eps = self.company.get_earnings_dates(limit=100).reset_index().drop_duplicates()
            for rep_date in self.report_dates:
                rep_ts = pd.Timestamp(rep_date, tz='America/New_York')
                closest = min(eps['Earnings Date'], key=lambda x: abs(x - rep_ts))
                self.eps_surprises[rep_date] = eps.loc[eps['Earnings Date'] == closest, 'Surprise(%)'].values[0]
        except Exception as e:
            logging.error(f"EPS surprise fetch failed for {self.ticker}: {e}")

    def compute_firm_size(self):
        try:
            shares_df = self.company.get_shares_full(start="2018-02-01", end=None)
            for rep_date in self.report_dates:
                ts = self._get_nearest_trading_day(rep_date)
                price = self.hist.loc[ts]['Open']
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
