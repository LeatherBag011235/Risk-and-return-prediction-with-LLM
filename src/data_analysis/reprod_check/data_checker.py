import polars as pl
import pandas as pd
from datetime import datetime
from psycopg2.extras import execute_batch
import psycopg2
from tqdm import tqdm
import numpy as np
import logging
import re
import os

from src.data_collection.consts import  DB_PARAMS
from src.data_collection.targets_calculation.logging_config import logger


class ColumnRestorer:
    def __init__(self, db_params, root_dir: str):
        self.conn = psycopg2.connect(**db_params)
        self.root_dir = root_dir
        self.dict_for_anal = self._load_old_data()
        self.df = self._prepare_df()


    def _load_old_data(self) -> dict[str, pl.DataFrame]:
        
        data = {}
        pattern = re.compile(r'.*(?=\.parquet)')
        for file in os.listdir(self.root_dir):
            match = pattern.search(file)
            if not match:
                continue
            ticker = match.group(0)
            file_path = os.path.join(self.root_dir, file)
            df = pl.read_parquet(file_path)
            data[ticker] = df
        logger.info(f"✅ Loaded {len(data)} tickers from {self.root_dir}")
        return data

    def _prepare_df(self) -> pd.DataFrame:
        total_df = pl.DataFrame()

        for ticker, df in self.dict_for_anal.items():
            row_names = df.select(pl.col("row_names")).to_series().to_list()
            if "doc_length" in row_names:
                df = df.filter(pl.col("row_names") != "doc_length")
                row_names.remove("doc_length")
    
            df_no_index = df.select(pl.exclude("row_names"))
    
            date_cols = [datetime.strptime(col, "%Y-%m-%d").date() for col in df_no_index.columns]

            df_transposed = df_no_index.transpose(include_header=False)
        
            df_transposed.columns = row_names  # <- assign proper column names

            df_transposed = df_transposed.with_columns([
                pl.Series("date", date_cols),
                pl.lit(ticker).alias("ticker")
            ])

            df_final = df_transposed.select(["ticker", "date"] + row_names)
            total_df.vstack(df_final, in_place=True)

        df_pd = total_df.to_pandas()
        df_pd["date"] = pd.to_datetime(df_pd["date"]).dt.date
        return df_pd.set_index(["ticker", "date"])

    def column_exists(self, cur, table: str, col: str) -> bool:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
            )
        """, (table, col))
        return cur.fetchone()[0]

    def update_column(self, old_col: str, new_col: str):
        print(self.df)
        if old_col not in self.df.columns:
            logger.error(f"❌ Column '{old_col}' not in restored DataFrame")
            return

        with self.conn, self.conn.cursor() as cur:
            if not self.column_exists(cur, 'reports_2', new_col):
                logger.error(f"❌ Column '{new_col}' does not exist in reports_2")
                return

            # Clear previous values
            logger.info(f"🧹 Nulling out column '{new_col}' in reports_2 before update...")
            cur.execute(f"UPDATE reports_2 SET {new_col} = NULL")

            match_count = 0
            missing_count = 0

            for (ticker, filed_date), row in tqdm(self.df.iterrows(), desc=f"Updating {new_col} from {old_col}"):
                value = row[old_col]

                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                cur.execute("""
                    UPDATE reports_2
                    SET {0} = %s
                    WHERE cik = (
                        SELECT cik FROM companies WHERE ticker = %s LIMIT 1
                    ) AND filed_date = %s
                """.format(new_col), (value, ticker, filed_date))

                if cur.rowcount > 0:
                    match_count += 1
                else:
                    missing_count += 1

            logger.info(f"✅ Finished updating '{new_col}' from '{old_col}'")
            logger.info(f"🔢 Matched rows: {match_count}, Skipped rows (no match): {missing_count}")
