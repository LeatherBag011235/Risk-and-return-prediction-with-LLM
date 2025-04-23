import multiprocessing as mp
import psycopg2
import pandas as pd
from datetime import date
from psycopg2.extras import execute_batch
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from tqdm import tqdm


from src.data_collection.targets_calculation.targets_parser_class import TargetsParser

class TargetExecutor:
    def __init__(  
            self, 
            api_key, 
            secret_key, 
            db_params, 
            pool_size=None
        ):

        self.api_key = api_key
        self.secret_key = secret_key
        self.db_params = db_params
        self.pool_size = pool_size or mp.cpu_count()

        self.expected_keys = [
            "two_day_r", "three_day_r", "four_day_r", "five_day_r", "six_day_r", "seven_day_r", "full_q_r",
            "two_day_e_r", "three_day_e_r", "four_day_e_r", "five_day_e_r", "six_day_e_r", "seven_day_e_r", "full_q_e_r",
            "two_day_abn_r", "three_day_abn_r", "four_day_abn_r", "five_day_abn_r", "six_day_abn_r", "seven_day_abn_r", "full_q_abn_r",
            "two_day_r_vol", "three_day_r_vol", "four_day_r_vol", "five_day_r_vol", "six_day_r_vol", "seven_day_r_vol", "full_q_r_vol"
        ]
        self.cols = ",".join(["report_id"] + self.expected_keys)
        self.placeholders = ",".join(["%s"] * (1 + len(self.expected_keys)))

    def get_db_conn(self):
        return psycopg2.connect(**self.db_params)

    def fetch_companies(self):
        with self.get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT cik, ticker FROM companies WHERE ticker IS NOT NULL;")
                return cur.fetchall()

    def fetch_report_dates(self, cik):
        with self.get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT filed_date FROM reports WHERE cik = %s ORDER BY filed_date;", (cik,))
                return [r[0] for r in cur.fetchall()]

    def get_snp500_daily(self):
        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        start_date = pd.to_datetime("2017-01-01").date()
        end_date = date.today()

        request = StockBarsRequest(
            symbol_or_symbols=["SPY"],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
        )
        df = client.get_stock_bars(request).df
        df = df[df.index.get_level_values(0) == "SPY"]
        df.index = df.index.droplevel(0)
        df = df.sort_index()

        daily_vwap = df['vwap'].resample('1D').last().dropna()
        snp500_daily = pd.DataFrame({"Close": daily_vwap})
        if snp500_daily.index.tz is None:
            snp500_daily.index = snp500_daily.index.tz_localize("UTC")
        snp500_daily.index = snp500_daily.index.tz_convert("America/New_York").normalize()

        return snp500_daily

    def worker(self, args):
        cik, ticker, snp500_daily = args
        try:
            report_dates = self.fetch_report_dates(cik)
            parser = TargetsParser(ticker, report_dates, snp500_daily, self.api_key, self.secret_key,)

            parser.compute_price_metrics()
            parser.compute_eps_surprise()
            parser.compute_firm_size()

            company_updates = (
                parser.company.info.get("sector"),
                parser.factor_model.params["const"],
                parser.const_significance["0.05"],
                parser.const_significance["0.01"],
                parser.const_significance["0.001"],
                cik
            )

            report_updates = []
            target_rows = []

            for date in report_dates:
                eps_size = parser.get_eps_and_size(date)
                report_updates.append((eps_size["eps_surprise"], eps_size["f_size"], cik, date))

                row = parser.assemble_target_row(date)
                if row:
                    target_rows.append((cik, date, *row.values()))

            return (company_updates, report_updates, target_rows)

        except Exception as e:
            print(f"❌ Error for {ticker}: {e}")
            return None

    def insert_results(self, results):
        with self.get_db_conn() as conn:
            with conn.cursor() as cur:
                for r in results:
                    if r is None:
                        continue

                    company_data, report_data, target_data = r

                    cur.execute("""
                        UPDATE companies SET sector=%s, alpha=%s, sig_005=%s, sig_001=%s, sig_0001=%s WHERE cik=%s
                    """, company_data)

                    execute_batch(cur, """
                        UPDATE reports SET eps_surprise=%s, firm_size=%s WHERE cik=%s AND filed_date=%s
                    """, report_data)

                    for row in target_data:
                        cik, filed_date, *metrics = row
                        cur.execute("SELECT id FROM reports WHERE cik=%s AND filed_date=%s", (cik, filed_date))
                        report_id = cur.fetchone()
                        if not report_id:
                            continue
                        if len(metrics) != len(self.expected_keys):
                            raise ValueError(f"Mismatch in metrics count for {cik} on {filed_date}")
                        cur.execute(
                            f"INSERT INTO targets ({self.cols}) VALUES ({self.placeholders})",
                            (report_id[0], *metrics)
                        )
            conn.commit()

    def run(self):
        snp500_daily = self.get_snp500_daily()
        all_companies = self.fetch_companies()
    
        with mp.Pool(self.pool_size) as pool:
            tasks = [(cik, ticker, snp500_daily) for cik, ticker in all_companies]
            results = list(tqdm(pool.imap_unordered(self.worker, tasks), total=len(tasks), desc="Processing companies"))
    
        self.insert_results(results)