import psycopg2
import pandas as pd
from datetime import datetime, date
from typing import Any

class DataFetcher:
    """
    DataFetcher class for retrieving and preparing financial report data
    from a PostgreSQL database for regression analysis.

    Responsibilities:
    - Connect to the database
    - Fetch regressors from the reports table
    - Fetch targets from the targets table
    - Optionally filter by company characteristics from the companies table
    - Expand list-type regressors into multiple columns
    - Prepare data structure suitable for fixed effects regression
    """

    market_cap_labels = ["small", "mid", "large"]
    market_cap_bins = [-float("inf"), 2e9, 1e10, float("inf")]

    def __init__(
        self,
        db_params: dict[str, Any],
        reports_table: str = 'reports',
        targets_table: str = 'targets',
        companies_table: str = 'companies'
    ) -> None:
        """
        Initialize DataFetcher with database connection parameters and table names.
        Also fetch and print available regressors and sectors.

        Args:
            db_params: Dictionary with DB connection parameters (dbname, user, password, host, port)
            reports_table: Name of the reports table
            targets_table: Name of the targets table
            companies_table: Name of the companies table
        """
        self.db_params = db_params
        self.reports_table = reports_table
        self.targets_table = targets_table
        self.companies_table = companies_table

        self.available_regressors = self._fetch_available_regressors()
        print("Available regressors:")
        for reg in self.available_regressors:
            print(f" - {reg}")

        self.sectors = self._print_available_sectors()

    def get_db_conn(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(**self.db_params)

    def fetch_data(
        self,
        regressors: list[str] | None = None,
        company_filters: dict[str, Any] | None = None,
        report_filters: dict[str, Any] | None = None,
        prepare_fixed_effects: bool = False
    ) -> pd.DataFrame:
        
        reports_df = self._fetch_reports(regressors, report_filters)

        targets_df = self._fetch_targets()
        merged_df = reports_df.merge(targets_df, left_on='id', right_on='report_id', how='inner')
        merged_df.drop(columns='report_id', inplace=True)

        if company_filters or prepare_fixed_effects:
            merged_df = self._apply_company_filters(merged_df, company_filters or {})

        if prepare_fixed_effects:
            merged_df = self._prepare_fixed_effects(merged_df)

        return merged_df

    def fetch_reports_with_company_metadata(
        self,
        regressors: list[str] | None = None,
        company_filters: dict[str, Any] | None = None,
        report_filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        reports_df = self._fetch_reports(regressors, report_filters)
        return self._apply_company_filters(reports_df, company_filters or {})

    def derive_company_quarter_labels(
        self,
        reports_df: pd.DataFrame,
        report_id_col: str = "id",
        output_col: str = "company_quarter",
    ) -> pd.DataFrame:
        required_cols = {report_id_col, "cik", "filed_date"}
        missing_cols = required_cols.difference(reports_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for quarter labels: {sorted(missing_cols)}"
            )

        if reports_df.empty:
            return pd.DataFrame(columns=[report_id_col, output_col])

        labels_df = reports_df.copy()
        labels_df["filed_date"] = pd.to_datetime(labels_df["filed_date"])
        labels_df = labels_df.sort_values(["cik", "filed_date", report_id_col]).copy()
        labels_df["year"] = labels_df["filed_date"].dt.year
        labels_df["filing_rank"] = labels_df.groupby(["cik", "year"]).cumcount()
        labels_df = labels_df[
            labels_df["filing_rank"] >= (
                labels_df.groupby(["cik", "year"])["filing_rank"].transform("max") - 3
            )
        ].copy()
        labels_df[output_col] = labels_df.groupby(["cik", "year"]).cumcount() + 1
        return labels_df[[report_id_col, output_col]]

    def derive_company_quarter_labels_by_cycle(
        self,
        reports_df: pd.DataFrame,
        report_id_col: str = "id",
        output_col: str = "company_quarter",
    ) -> pd.DataFrame:
        required_cols = {report_id_col, "cik", "filed_date", "report_type"}
        missing_cols = required_cols.difference(reports_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for cycle-based quarter labels: {sorted(missing_cols)}"
            )

        if reports_df.empty:
            return pd.DataFrame(columns=[report_id_col, output_col])

        labels_df = reports_df.copy()
        labels_df["filed_date"] = pd.to_datetime(labels_df["filed_date"])
        labels_df = labels_df[
            labels_df["report_type"].isin(["10-K", "10-Q"]) & labels_df["filed_date"].notna()
        ].copy()
        labels_df = labels_df.sort_values(["cik", "filed_date", report_id_col]).copy()

        label_rows: list[dict[str, Any]] = []

        for _, company_df in labels_df.groupby("cik", sort=False):
            company_df = company_df.reset_index(drop=True)
            segment_start = 0

            annual_positions = company_df.index[company_df["report_type"] == "10-K"].tolist()
            for annual_pos in annual_positions:
                segment_df = company_df.iloc[segment_start:annual_pos + 1].copy()
                if not segment_df.empty:
                    labeled_segment = segment_df.tail(4).copy()
                    if labeled_segment.iloc[-1]["report_type"] == "10-K":
                        quarter_labels = list(range(5 - len(labeled_segment), 5))
                        if len(labeled_segment) == 4:
                            quarter_labels = [1, 2, 3, 4]
                        labeled_segment[output_col] = quarter_labels
                        label_rows.extend(
                            labeled_segment[[report_id_col, output_col]].to_dict("records")
                        )
                segment_start = annual_pos + 1

            tail_df = company_df.iloc[segment_start:].copy()
            if not tail_df.empty and tail_df["report_type"].eq("10-Q").all():
                tail_length = min(len(tail_df), 3)
                labeled_tail = tail_df.head(tail_length).copy()
                labeled_tail[output_col] = list(range(1, tail_length + 1))
                label_rows.extend(
                    labeled_tail[[report_id_col, output_col]].to_dict("records")
                )

        if not label_rows:
            return pd.DataFrame(columns=[report_id_col, output_col])

        result_df = pd.DataFrame(label_rows).drop_duplicates(subset=[report_id_col], keep="last")
        return result_df[[report_id_col, output_col]]

    @classmethod
    def bucket_market_cap(cls, f_size: float) -> str | None:
        if pd.isna(f_size):
            return None

        return pd.cut(
            pd.Series([f_size]),
            bins=cls.market_cap_bins,
            labels=cls.market_cap_labels,
            right=False,
        ).iloc[0]

    def add_market_cap_bucket(
        self,
        reports_df: pd.DataFrame,
        source_col: str = "f_size",
        output_col: str = "market_cap_label",
    ) -> pd.DataFrame:
        if source_col not in reports_df.columns:
            raise ValueError(f"Column '{source_col}' not found in reports DataFrame.")

        labeled_df = reports_df.copy()
        labeled_df[output_col] = labeled_df[source_col].apply(self.bucket_market_cap)
        labeled_df[output_col] = labeled_df[output_col].astype("object")
        return labeled_df

    def _print_available_sectors(self) -> None:
        with self.get_db_conn() as conn:
            query = f"""
                SELECT sector, COUNT(*) AS count
                FROM {self.companies_table}
                WHERE sector IS NOT NULL
                GROUP BY sector
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn)

        sectors = {}
        print("Available sectors:")
        for _, row in df.iterrows():
            print(f" - {row['sector']} ({row['count']})")
            sectors[row['sector']] = row['count']

        return sectors

    def _fetch_available_regressors(self) -> list[str]:
        with self.get_db_conn() as conn:
            query = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{self.reports_table}'
                  AND column_name NOT IN ('id', 'cik', 'filed_date', 'report_type', 'url', 'raw_text', 'extracted_data', 'created_at')
                ORDER BY column_name;
            """
            df = pd.read_sql_query(query, conn)
            return df['column_name'].tolist()

    def _fetch_reports(
        self,
        regressors: list[str] | None,
        report_filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        with self.get_db_conn() as conn:
            columns = ['id', 'cik', 'filed_date', 'report_type']
            if regressors:
                columns += regressors
            else:
                columns += self.available_regressors

            cols_str = ", ".join(columns)
            where_clauses = [f"{reg} IS NOT NULL" for reg in regressors] if regressors else []
            if report_filters:
                for col, val in report_filters.items():
                    if isinstance(val, list):
                        values = ", ".join(f"'{item}'" for item in val)
                        where_clauses.append(f"{col} IN ({values})")
                    else:
                        where_clauses.append(f"{col} = '{val}'")
            where_sql = " AND ".join(where_clauses)
            if where_sql:
                query = f"SELECT {cols_str} FROM {self.reports_table} WHERE {where_sql}"
            else:
                query = f"SELECT {cols_str} FROM {self.reports_table}"

            df = pd.read_sql_query(query, conn)

        return self._expand_list_columns(df, regressors)

    def _apply_report_filters(
        self,
        reports_df: pd.DataFrame,
        report_filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not report_filters:
            return reports_df

        filtered_df = reports_df.copy()
        for col, val in report_filters.items():
            if col not in filtered_df.columns:
                raise ValueError(f"Filter column '{col}' not found in reports table.")
            if isinstance(val, list):
                filtered_df = filtered_df[filtered_df[col].isin(val)]
            else:
                filtered_df = filtered_df[filtered_df[col] == val]
        return filtered_df

    def _expand_list_columns(self, df: pd.DataFrame, regressors: list[str] | None) -> pd.DataFrame:
        if not regressors:
            regressors = self.available_regressors

        for reg in regressors:
            if reg not in df.columns:
                continue
            if df[reg].dropna().apply(lambda x: isinstance(x, (list, tuple))).any():
                non_null_lists = df[reg].dropna()
                lengths = non_null_lists.apply(len)
                if lengths.nunique() != 1:
                    raise ValueError(f"Regressor '{reg}' contains lists of different lengths!")

                list_len = lengths.iloc[0]
                print(f"Expanding list regressor '{reg}' into {list_len} columns...")

                expanded_cols = pd.DataFrame(non_null_lists.tolist(), index=non_null_lists.index)
                expanded_cols.columns = [f"segment_{i+1}" for i in range(list_len)]

                df = pd.concat([df.drop(columns=[reg]), expanded_cols], axis=1)

        return df

    def _fetch_targets(self) -> pd.DataFrame:
        with self.get_db_conn() as conn:
            query = f"SELECT * FROM {self.targets_table}"
            return pd.read_sql_query(query, conn)

    def _apply_company_filters(self, merged_df: pd.DataFrame, company_filters: dict[str, Any]) -> pd.DataFrame:
        with self.get_db_conn() as conn:
            query = f"""
                SELECT cik, ticker, sector, industry, alpha, sig_005, sig_001, sig_0001
                FROM {self.companies_table}
            """
            companies_df = pd.read_sql_query(query, conn)

        merged_df = merged_df.merge(companies_df, on='cik', how='left')

        for col, val in company_filters.items():
            if isinstance(val, list):
                merged_df = merged_df[merged_df[col].isin(val)]
            else:
                merged_df = merged_df[merged_df[col] == val]

        return merged_df

    def _prepare_fixed_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('filed_date')
        df['year'] = pd.to_datetime(df['filed_date']).dt.year
        df['filing_rank'] = df.groupby(['cik', 'year']).cumcount()
        df = df[df['filing_rank'] >= (df.groupby(['cik', 'year'])['filing_rank'].transform('max') - 3)]
        df['quarter_rank'] = df.groupby(['cik', 'year']).cumcount() + 1

        overfilled = df[df['quarter_rank'] > 4]
        if not overfilled.empty:
            problem_rows = (
                overfilled.groupby(['cik', 'year'])
                .size()
                .reset_index(name='num_reports')
                .to_string(index=False)
            )
            raise ValueError(f"Detected >4 filings per year for some companies:\n{problem_rows}")

        df['date'] = (df['year'] + df['quarter_rank'] * 0.1).round(1)
        df['company'] = df['ticker']
        df.set_index(['company', 'date'], inplace=True)

        cols_to_drop = [
            col for col in [
                'id', 'cik', 'ticker', 'filed_date',
                'sector', 'industry', 'alpha',
                'sig_005', 'sig_001', 'sig_0001',
                'year', 'quarter_rank', 'filing_rank'
            ] if col in df.columns
        ]
        df.drop(columns=cols_to_drop, inplace=True)
        df.sort_index(inplace=True)
        return df
