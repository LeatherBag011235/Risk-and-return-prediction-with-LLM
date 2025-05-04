import psycopg2
import pandas as pd
from datetime import datetime
from typing import Any
from datetime import datetime, date

class DataFetcher:
    """
    DataFetcher class for retrieving and preparing financial report data
    from a PostgreSQL database for regression analysis.

    Responsibilities:
    - Connect to the database
    - Fetch regressors from the 'reports' table
    - Fetch targets from the 'targets' table
    - Optionally filter by company characteristics from the 'companies' table
    - Expand list-type regressors into multiple columns
    - Prepare data structure suitable for fixed effects regression
    """

    def __init__(self, db_params: dict[str, Any]) -> None:
        """
        Initialize DataFetcher with database connection parameters.
        Also fetch and print available regressors.

        Args:
            db_params: Dictionary with DB connection parameters (dbname, user, password, host, port)
        """
        self.db_params = db_params
        self.available_regressors = self._fetch_available_regressors()
        print("Available regressors:")
        for reg in self.available_regressors:
            print(f" - {reg}")
        
        self.sectors = self._print_available_sectors()

    def get_db_conn(self) -> psycopg2.extensions.connection:
        """
        Open a new database connection.

        Returns:
            A psycopg2 database connection object
        """
        return psycopg2.connect(**self.db_params)
    
    def fetch_data(
        self,
        regressors: list[str] | None = None,
        company_filters: dict[str, Any] | None = None,
        report_filters: dict[str, Any] | None = None,
        prepare_fixed_effects: bool = False
    ) -> pd.DataFrame:
        """
        Fetch reports and targets from the database, merge, and optionally filter and reformat.

        Args:
            regressors: List of regressor column names to select (default all available)
            company_filters: Dict of filters from companies table (e.g., {'sector': 'Tech'})
            report_filters: Dict of filters from reports table (e.g., {'report_type': '10-Q'})
            prepare_fixed_effects: Whether to format for fixed effects regression

        Returns:
            A pandas DataFrame ready for analysis
        """
        reports_df = self._fetch_reports(regressors)

        # ✅ Apply filters to reports
        if report_filters:
            for col, val in report_filters.items():
                if col not in reports_df.columns:
                    raise ValueError(f"Filter column '{col}' not found in reports table.")
                if isinstance(val, list):
                    reports_df = reports_df[reports_df[col].isin(val)]
                else:
                    reports_df = reports_df[reports_df[col] == val]

        targets_df = self._fetch_targets()
        merged_df = reports_df.merge(targets_df, left_on='id', right_on='report_id', how='inner')
        merged_df.drop(columns='report_id', inplace=True)

        if company_filters or prepare_fixed_effects:
            merged_df = self._apply_company_filters(merged_df, company_filters or {})

        if prepare_fixed_effects:
            merged_df = self._prepare_fixed_effects(merged_df)

        return merged_df
    
    def _print_available_sectors(self) -> None:
        """
        Query and print all distinct sectors in the companies table, with row counts.

        Example output:
            Available sectors:
             - Technology (54)
             - Healthcare (38)
             ...
        """
        with self.get_db_conn() as conn:
            query = """
                SELECT sector, COUNT(*) AS count
                FROM companies
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
        """
        Fetch all column names from the 'reports' table excluding meta-columns.

        Returns:
            A list of available regressor column names
        """
        with self.get_db_conn() as conn:
            query = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'reports'
                  AND column_name NOT IN ('id', 'cik', 'filed_date', 'report_type', 'url', 'raw_text', 'extracted_data', 'created_at')
                ORDER BY column_name;
            """
            df = pd.read_sql_query(query, conn)
            return df['column_name'].tolist()
    
    def _fetch_reports(self, regressors: list[str] | None) -> pd.DataFrame:
        """
        Fetch selected regressors from the 'reports' table, keeping only non-null rows.
        Also expand list-type regressors into separate columns if necessary.

        Args:
            regressors: List of regressor column names to fetch

        Returns:
            A pandas DataFrame containing selected regressors
        """
        with self.get_db_conn() as conn:
            columns = ['id', 'cik', 'filed_date', 'report_type']
            if regressors:
                columns += regressors
            else:
                columns += self.available_regressors
            
            cols_str = ", ".join(columns)
            where_clauses = []
            if regressors:
                for reg in regressors:
                    where_clauses.append(f"{reg} IS NOT NULL")
            where_sql = " AND ".join(where_clauses)
            if where_sql:
                query = f"SELECT {cols_str} FROM reports WHERE {where_sql}"
            else:
                query = f"SELECT {cols_str} FROM reports"
            
            df = pd.read_sql_query(query, conn)
        
        df = self._expand_list_columns(df, regressors)
        return df
    
    def _expand_list_columns(self, df: pd.DataFrame, regressors: list[str] | None) -> pd.DataFrame:
        """
        Expand list-type regressor columns into multiple scalar columns.

        Args:
            df: DataFrame containing the reports
            regressors: List of regressor names to check and expand

        Returns:
            A DataFrame with expanded columns
        """
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
        """
        Fetch all targets from the 'targets' table.

        Returns:
            A pandas DataFrame containing targets
        """
        with self.get_db_conn() as conn:
            query = "SELECT * FROM targets"
            return pd.read_sql_query(query, conn)
    
    def _apply_company_filters(self, merged_df: pd.DataFrame, company_filters: dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters based on company characteristics.

        Args:
            merged_df: Merged DataFrame of reports and targets
            company_filters: Dictionary specifying filtering conditions

        Returns:
            A filtered pandas DataFrame
        """
        with self.get_db_conn() as conn:
            query = "SELECT cik, ticker, sector, industry, alpha, sig_005, sig_001, sig_0001 FROM companies"
            companies_df = pd.read_sql_query(query, conn)
        
        merged_df = merged_df.merge(companies_df, on='cik', how='left')
        
        for col, val in company_filters.items():
            if isinstance(val, list):
                merged_df = merged_df[merged_df[col].isin(val)]
            else:
                merged_df = merged_df[merged_df[col] == val]
        
        return merged_df
    
    def _prepare_fixed_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns quarter rank based on filing date within each company-year,
        constructs `company` and `date` (as year.quarter float),
        validates max 4 filings per year per company,
        sets index, and drops unused columns.
        """
        # Step 1: Sort to ensure proper order for ranking
        df = df.sort_values('filed_date')
        df['year'] = pd.to_datetime(df['filed_date']).dt.year
        
        # Step 2: Assign per-year rank within each company
        df['filing_rank'] = df.groupby(['cik', 'year']).cumcount()
        
        # Step 3: Keep only rows with rank 1 to 4 (i.e., drop rank 0 if there are 5 filings)
        # This keeps latest 4 filings per year
        df = df[df['filing_rank'] >= (df.groupby(['cik', 'year'])['filing_rank'].transform('max') - 3)]
        
        # Step 4: Recompute quarter rank as 1–4 (since we dropped the earliest)
        df['quarter_rank'] = df.groupby(['cik', 'year']).cumcount() + 1

    
        # Check: no (cik, year) has more than 4 filings
        overfilled = df[df['quarter_rank'] > 4]
        if not overfilled.empty:
            problem_rows = (
                overfilled.groupby(['cik', 'year'])
                .size()
                .reset_index(name='num_reports')
                .to_string(index=False)
            )
            raise ValueError(f"Detected >4 filings per year for some companies:\n{problem_rows}")
    
        # Assign float quarter (e.g., 2022.1, 2022.2, ...)
        df['date'] = (df['year'] + df['quarter_rank'] * 0.1).round(1)
        df['company'] = df['ticker']
    
        df.set_index(['company', 'date'], inplace=True)
    
        cols_to_drop = [
            col for col in [
                'id', 'cik', 'ticker', 'filed_date',
                'sector', 'industry', 'alpha',
                'sig_005', 'sig_001', 'sig_0001',
                'year', 'quarter_rank'
            ] if col in df.columns
        ]
        df.drop(columns=cols_to_drop, inplace=True)
    
        df.sort_index(inplace=True)
        return df
