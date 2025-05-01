from ast import parse
import multiprocessing as mp
import psycopg2
import pandas as pd
from datetime import date
from psycopg2.extras import execute_batch
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from tqdm import tqdm

from src.data_collection.targets_calculation.target_parser_class import TargetsParser
from src.data_collection.targets_calculation.logging_config import logger

class TargetExecutor:
    """
    TargetExecutor orchestrates the full pipeline for collecting, computing, and inserting
    earnings-related target metrics for companies into a PostgreSQL database.

    Steps handled:
    - Fetch company and report data
    - Download stock and market data
    - Compute returns, volatility, abnormal returns, EPS surprises, and firm size
    - Update the `companies`, `reports`, and `targets` tables
    - Parallelize across companies using multiprocessing
    """

    def __init__(self, api_key: str, secret_key: str, db_params: dict, pool_size: int | None = None):
        """
        Initialize the TargetExecutor.

        Args:
            api_key: Alpaca API key.
            secret_key: Alpaca secret key.
            db_params: Dictionary with database connection parameters.
            pool_size: Number of processes to use (defaults to CPU count).
        """
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

    def get_db_conn(self) -> psycopg2.extensions.connection:
        """
        Open a new database connection.

        Returns:
            psycopg2 connection
        """
        return psycopg2.connect(**self.db_params)

    def fetch_companies(self) -> list[tuple[str, str]]:
        """
        Fetch companies that meet the following criteria:
        - Have at least one report with a non-null full_list_default_verbolizer.
        - Do not already have alpha in companies table AND 2-day excess and abnormal returns in targets table.

        Returns:
            List of (cik, ticker) pairs.
        """
        with self.get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT c.cik, c.ticker
                FROM companies c
                JOIN reports r ON c.cik = r.cik
                WHERE r.full_list_default_verbolizer IS NOT NULL
                  AND (
                    c.alpha IS NULL
                    OR NOT EXISTS (
                        SELECT 1
                        FROM reports r2
                        JOIN targets t ON r2.id = t.report_id
                        WHERE r2.cik = c.cik
                          AND t.two_day_e_r IS NOT NULL
                          AND t.two_day_abn_r IS NOT NULL
                    )
                  );
            """)
            return cur.fetchall()

    def fetch_report_dates(self, cik: str) -> list[date]:
        """
        Fetch filing dates for a specific company (CIK).

        Args:
            cik: Company identifier.

        Returns:
            List of filing dates.
        """
        with self.get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT filed_date FROM reports WHERE cik = %s ORDER BY filed_date;", (cik,))
            return [r[0] for r in cur.fetchall()]

    def get_snp500_hourly(self) -> pd.DataFrame:
        """
        Fetch daily SPY VWAP data.

        Returns:
            Daily VWAP dataframe, timezone normalized.
        """
        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        start_date = pd.to_datetime("2017-01-01").date()
        end_date = date.today()

        request = StockBarsRequest(
            symbol_or_symbols=["SPY"],
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
        )

        df = client.get_stock_bars(request).df
        df = df[df.index.get_level_values(0) == "SPY"]
        df.index = df.index.droplevel(0)
        df = df.sort_index()

        return df

    def worker(self, args: tuple[str, str, pd.DataFrame]) -> tuple[tuple, list[tuple], list[tuple]] | None:
        """
        Process a single company.

        Args:
            args: (cik, ticker, snp500_daily).

        Returns:
            (company_updates, report_updates, target_rows) or None if error.
        """
        cik, ticker, snp500_hourly = args
        try:
            report_dates = self.fetch_report_dates(cik)
    
            assert isinstance(snp500_hourly.index, pd.DatetimeIndex), "Not a DatetimeIndex"
            assert snp500_hourly.index.is_monotonic_increasing, "Index not sorted"


            parser = TargetsParser(ticker, report_dates, snp500_hourly, self.api_key, self.secret_key,)

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
            logger.debug(f"company updates: {company_updates}")

            report_updates = []
            target_rows = []

            for date in report_dates:
                eps_size = parser.get_eps_and_size(date)
                report_updates.append((eps_size["eps_surprise"], eps_size["f_size"], cik, date))

                row = parser.assemble_target_row(date)
                if row:
                    target_rows.append((cik, date, *row.values()))

            return company_updates, report_updates, target_rows

        except Exception as e:
            print(f"❌ Error for {ticker}: {e}")
            return None

    def insert_results(self, results: list[tuple[tuple, list[tuple], list[tuple]]]) -> None:
        """
        Insert all fetched and computed results into the database.

        Args:
            results: Results returned by worker processes.
        """
        with self.get_db_conn() as conn, conn.cursor() as cur:
            for r in results:
                if r is None:
                    logger.warning(f"results in insert_results are None")
                    continue

                company_data, report_data, target_data = r
                    

                cur.execute("""
                    UPDATE companies SET sector=%s, alpha=%s, sig_005=%s, sig_001=%s, sig_0001=%s WHERE cik=%s
                """, company_data)

                execute_batch(cur, """
                    UPDATE reports SET eps_surprise=%s, f_size=%s WHERE cik=%s AND filed_date=%s
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
                        f"""
                        INSERT INTO targets ({self.cols})
                        VALUES ({self.placeholders})
                        ON CONFLICT (report_id) DO UPDATE SET
                        {', '.join([f"{col}=EXCLUDED.{col}" for col in self.expected_keys])}
                        """,
                        (report_id[0], *metrics)
                    )

            conn.commit()

    def run(self, single_ticker: str | None = None) -> None:
        """
        Launch the multiprocessing pipeline to process all companies or a specific ticker.

        Args:
            single_ticker: If provided, only this ticker will be processed.
        """
        snp500_hourly = self.get_snp500_hourly()

        if single_ticker is not None:
            with self.get_db_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT cik FROM companies WHERE ticker = %s", (single_ticker,))
                res = cur.fetchone()
                if not res:
                    logger.warning(f"Ticker {single_ticker} not found in database.")
                    return
                cik = res[0]
                logger.info(f"Processing single ticker: {single_ticker} (CIK: {cik})")
                result = self.worker((cik, single_ticker, snp500_hourly))
                if result:
                    self.insert_results([result])
        else:
            all_companies = self.fetch_companies()
            with mp.Pool(self.pool_size) as pool:
                tasks = [(cik, ticker, snp500_hourly) for cik, ticker in all_companies]
                for result in tqdm(pool.imap_unordered(self.worker, tasks), total=len(tasks), desc="Processing companies"):
                    if result:
                        self.insert_results([result])
