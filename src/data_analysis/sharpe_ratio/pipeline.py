from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data_analysis.data_fetcher.data_fetcher_class import DataFetcher

SIGNAL_COLUMN = "max_abs_shrink"
REPORT_TYPES = ["10-K", "10-Q"]
LEG_FRACTION = 0.2
WEEKLY_HORIZON_TRADING_DAYS = 7
PRICE_CHUNK_SIZE = 50
PRICE_RETRY_CHUNK_SIZE = 10
PRICE_RETRY_PASSES = 1
PRICE_BUFFER_DAYS = 10
YF_TIMEOUT = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PRICE_CACHE_FILE = OUTPUT_DIR / "adjusted_closes.csv"
PRICE_CACHE_TICKERS_FILE = OUTPUT_DIR / "adjusted_close_attempted_tickers.csv"


def build_signal_panel(fetcher: DataFetcher) -> pd.DataFrame:
    """Fetch one investable sentiment signal per ticker-calendar-quarter."""
    reports_df = fetcher.fetch_reports_with_company_metadata(
        regressors=[SIGNAL_COLUMN],
        report_filters={"report_type": REPORT_TYPES},
    )

    quarter_labels = fetcher.derive_company_quarter_labels_by_cycle(
        reports_df,
        report_id_col="id",
        output_col="company_quarter",
    )

    panel = reports_df.merge(quarter_labels, on="id", how="inner")
    panel = panel[panel["ticker"].notna()].copy()
    panel["filed_date"] = pd.to_datetime(panel["filed_date"])
    panel["calendar_quarter"] = panel["filed_date"].dt.to_period("Q").astype(str)

    panel = panel.sort_values(["ticker", "calendar_quarter", "filed_date", "id"])
    panel = panel.drop_duplicates(["ticker", "calendar_quarter"], keep="last")

    return panel


def build_formation_dates(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """Use the latest filing date in each calendar quarter as formation date."""
    formation_dates = (
        signal_panel.groupby("calendar_quarter", as_index=False)
        .agg(
            formation_date=("filed_date", "max"),
            first_filing_date=("filed_date", "min"),
            filings=("id", "count"),
            tickers=("ticker", "nunique"),
        )
        .sort_values("calendar_quarter")
    )

    formation_dates["next_formation_date"] = formation_dates["formation_date"].shift(-1)
    return formation_dates


def build_portfolio_members(
    signal_panel: pd.DataFrame,
    formation_dates: pd.DataFrame,
    leg_fraction: float = LEG_FRACTION,
) -> pd.DataFrame:
    """Sort each quarter by signal and assign top/bottom equal-weighted legs."""
    if not 0 < leg_fraction < 0.5:
        raise ValueError("leg_fraction must be between 0 and 0.5.")

    formation_lookup = formation_dates.set_index("calendar_quarter")[
        ["formation_date", "next_formation_date"]
    ]
    member_rows = []

    for calendar_quarter, quarter_df in signal_panel.groupby("calendar_quarter", sort=True):
        quarter_df = quarter_df.dropna(subset=[SIGNAL_COLUMN, "ticker"]).copy()
        quarter_df = quarter_df.sort_values(
            [SIGNAL_COLUMN, "ticker", "filed_date", "id"],
            ascending=[True, True, True, True],
        )

        leg_size = int(len(quarter_df) * leg_fraction)
        if leg_size < 1:
            continue

        short_leg = quarter_df.head(leg_size).copy()
        long_leg = quarter_df.tail(leg_size).copy()

        for side, leg_df, signed_weight in [
            ("short", short_leg, -1.0 / leg_size),
            ("long", long_leg, 1.0 / leg_size),
        ]:
            leg_df = leg_df.assign(
                side=side,
                signed_weight=signed_weight,
                leg_weight=1.0 / leg_size,
                formation_date=formation_lookup.loc[calendar_quarter, "formation_date"],
                next_formation_date=formation_lookup.loc[
                    calendar_quarter, "next_formation_date"
                ],
            )
            member_rows.append(leg_df)

    if not member_rows:
        return pd.DataFrame()

    members = pd.concat(member_rows, ignore_index=True)
    return members[
        [
            "calendar_quarter",
            "formation_date",
            "next_formation_date",
            "side",
            "ticker",
            "cik",
            "filed_date",
            "report_type",
            SIGNAL_COLUMN,
            "leg_weight",
            "signed_weight",
        ]
    ].sort_values(["calendar_quarter", "side", SIGNAL_COLUMN, "ticker"])


def build_portfolio_summary(portfolio_members: pd.DataFrame) -> pd.DataFrame:
    """Compact quarter-level view of long and short members."""
    if portfolio_members.empty:
        return pd.DataFrame()

    summary = (
        portfolio_members.groupby(["calendar_quarter", "formation_date", "side"])
        .agg(
            names=("ticker", "count"),
            min_signal=(SIGNAL_COLUMN, "min"),
            median_signal=(SIGNAL_COLUMN, "median"),
            max_signal=(SIGNAL_COLUMN, "max"),
        )
        .reset_index()
        .sort_values(["calendar_quarter", "side"])
    )
    return summary


def to_yahoo_ticker(ticker: str) -> str:
    """Convert database tickers to Yahoo Finance symbols."""
    return ticker.replace(".", "-")


def _download_close_chunk(
    tickers: list[str],
    start: str,
    end: str,
    chunk_label: str,
) -> pd.DataFrame:
    yahoo_tickers = [to_yahoo_ticker(ticker) for ticker in tickers]
    yahoo_to_original = dict(zip(yahoo_tickers, tickers))

    print(f"Downloading prices {chunk_label}: {len(tickers)} tickers ({start} to {end})")
    raw_prices = yf.download(
        tickers=yahoo_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        threads=True,
        progress=False,
        group_by="column",
        timeout=YF_TIMEOUT,
    )

    if raw_prices.empty:
        print(f"  {chunk_label} returned no data")
        return pd.DataFrame()

    if isinstance(raw_prices.columns, pd.MultiIndex):
        if "Close" not in raw_prices.columns.get_level_values(0):
            print(f"  {chunk_label} has no Close columns")
            return pd.DataFrame()
        closes = raw_prices["Close"].copy()
    else:
        closes = raw_prices[["Close"]].copy()
        closes.columns = yahoo_tickers

    closes = closes.rename(columns=yahoo_to_original)
    closes.index = pd.to_datetime(closes.index).normalize()
    returned = int(closes.notna().any(axis=0).sum())
    print(f"  returned {returned}/{len(tickers)} tickers")
    return closes


def load_cached_adjusted_closes(cache_file: Path = PRICE_CACHE_FILE) -> pd.DataFrame:
    """Load cached adjusted closes, if available."""
    if not cache_file.exists():
        return pd.DataFrame()

    prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index).normalize()
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
    return prices


def load_cached_price_tickers(cache_file: Path = PRICE_CACHE_TICKERS_FILE) -> set[str]:
    """Load the ticker universe already attempted for the price cache."""
    if not cache_file.exists():
        return set()

    tickers = pd.read_csv(cache_file)["ticker"].dropna().astype(str)
    return set(tickers)


def save_price_cache(
    prices: pd.DataFrame,
    attempted_tickers: set[str] | list[str] | None = None,
    cache_file: Path = PRICE_CACHE_FILE,
    ticker_cache_file: Path = PRICE_CACHE_TICKERS_FILE,
) -> None:
    """Persist adjusted closes and the ticker universe attempted for reuse."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.sort_index().to_csv(cache_file, index_label="date")

    if attempted_tickers is not None:
        pd.DataFrame({"ticker": sorted(set(attempted_tickers))}).to_csv(
            ticker_cache_file,
            index=False,
        )


def price_cache_covers_request(
    prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> bool:
    """Check whether requested dates can map to available cached trading dates."""
    if prices.empty:
        return False

    price_index = pd.DatetimeIndex(prices.index).sort_values()
    for requested_date in pd.to_datetime([start_date, end_date]).normalize():
        mapped_position = price_index.searchsorted(requested_date, side="left")
        if mapped_position >= len(price_index):
            return False

        mapped_date = price_index[mapped_position]
        if mapped_date > requested_date + pd.Timedelta(days=PRICE_BUFFER_DAYS):
            return False

    return True


def _download_close_chunks(
    tickers: list[str],
    start: str,
    end: str,
    chunk_size: int,
    chunk_prefix: str,
) -> pd.DataFrame:
    """Download adjusted closes for a ticker list using fixed-size chunks."""
    price_frames = []
    for chunk_number, start_index in enumerate(
        range(0, len(tickers), chunk_size),
        start=1,
    ):
        chunk = tickers[start_index : start_index + chunk_size]
        closes = _download_close_chunk(
            chunk,
            start,
            end,
            f"{chunk_prefix} {chunk_number}",
        )
        if not closes.empty:
            price_frames.append(closes)

    if not price_frames:
        return pd.DataFrame()

    prices = pd.concat(price_frames, axis=1).sort_index()
    return prices.loc[:, ~prices.columns.duplicated(keep="last")]


def download_adjusted_closes(
    tickers: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    chunk_size: int = PRICE_CHUNK_SIZE,
    cache_file: Path = PRICE_CACHE_FILE,
    ticker_cache_file: Path = PRICE_CACHE_TICKERS_FILE,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Load cached adjusted closes when possible; otherwise download and cache them."""
    unique_tickers = sorted(set(tickers))
    requested_start = pd.to_datetime(start_date).normalize()
    requested_end = pd.to_datetime(end_date).normalize()
    start = requested_start.date().isoformat()
    end = (requested_end + pd.Timedelta(days=PRICE_BUFFER_DAYS)).date().isoformat()

    cached_prices = pd.DataFrame()
    attempted_tickers = set()
    tickers_to_download = unique_tickers

    if not refresh_cache:
        cached_prices = load_cached_adjusted_closes(cache_file)
        attempted_tickers = load_cached_price_tickers(ticker_cache_file)
        if price_cache_covers_request(cached_prices, requested_start, requested_end):
            missing_unattempted = sorted(
                set(unique_tickers) - set(cached_prices.columns) - attempted_tickers
            )
            if not missing_unattempted:
                print(
                    f"Using cached prices from {cache_file} "
                    f"({cached_prices.shape[0]:,} dates, {cached_prices.shape[1]:,} tickers)"
                )
                missing_cached_tickers = sorted(set(unique_tickers) - set(cached_prices.columns))
                if missing_cached_tickers:
                    print(
                        "  cache has no price columns for "
                        f"{len(missing_cached_tickers)} previously attempted tickers"
                    )
                return cached_prices

            print(
                f"Using cached prices for {len(unique_tickers) - len(missing_unattempted)}/"
                f"{len(unique_tickers)} requested tickers; downloading "
                f"{len(missing_unattempted)} new tickers"
            )
            tickers_to_download = missing_unattempted

    prices = _download_close_chunks(tickers_to_download, start, end, chunk_size, "chunk")
    if prices.empty and cached_prices.empty:
        raise RuntimeError("No price data returned from yfinance.")

    for retry_pass in range(1, PRICE_RETRY_PASSES + 1):
        returned_tickers = (
            set(prices.columns[prices.notna().any(axis=0)])
            if not prices.empty
            else set()
        )
        missing_tickers = sorted(set(tickers_to_download) - returned_tickers)
        if not missing_tickers:
            break

        print(f"Retry pass {retry_pass}: {len(missing_tickers)} missing tickers")
        retry_prices = _download_close_chunks(
            missing_tickers,
            start,
            end,
            PRICE_RETRY_CHUNK_SIZE,
            f"retry {retry_pass}.",
        )
        if not retry_prices.empty:
            prices = pd.concat([prices, retry_prices], axis=1).sort_index()
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    if not cached_prices.empty and price_cache_covers_request(
        cached_prices,
        requested_start,
        requested_end,
    ):
        prices = pd.concat([cached_prices, prices], axis=1).sort_index()
        prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]

    attempted_tickers = attempted_tickers | set(unique_tickers)
    save_price_cache(prices, attempted_tickers, cache_file, ticker_cache_file)
    print(f"Saved price cache to {cache_file}")
    return prices


def map_to_trading_dates(
    prices: pd.DataFrame,
    requested_dates: pd.Series | pd.Index,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map requested formation dates to the next available price date."""
    price_index = pd.DatetimeIndex(prices.index).sort_values()
    mapped_dates = {}

    for requested_date in pd.to_datetime(pd.Index(requested_dates).dropna().unique()).normalize():
        pos = price_index.searchsorted(requested_date, side="left")
        if pos < len(price_index):
            mapped_dates[requested_date] = price_index[pos]

    return mapped_dates


def compute_member_returns(
    portfolio_members: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Attach raw holding-period returns to sorted portfolio members."""
    members = portfolio_members.dropna(subset=["next_formation_date"]).copy()
    requested_dates = pd.concat([members["formation_date"], members["next_formation_date"]])
    trading_dates = map_to_trading_dates(prices, requested_dates)

    rows = []
    for row in members.itertuples(index=False):
        formation_date = pd.to_datetime(row.formation_date).normalize()
        next_formation_date = pd.to_datetime(row.next_formation_date).normalize()
        start_trade_date = trading_dates.get(formation_date)
        end_trade_date = trading_dates.get(next_formation_date)

        start_price = None
        end_price = None
        raw_return = None
        if (
            start_trade_date is not None
            and end_trade_date is not None
            and row.ticker in prices.columns
        ):
            start_price = prices.at[start_trade_date, row.ticker]
            end_price = prices.at[end_trade_date, row.ticker]
            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                raw_return = end_price / start_price - 1.0

        rows.append(
            {
                "calendar_quarter": row.calendar_quarter,
                "ticker": row.ticker,
                "side": row.side,
                "formation_date": formation_date,
                "next_formation_date": next_formation_date,
                "start_trade_date": start_trade_date,
                "end_trade_date": end_trade_date,
                SIGNAL_COLUMN: getattr(row, SIGNAL_COLUMN),
                "leg_weight": row.leg_weight,
                "signed_weight": row.signed_weight,
                "start_price": start_price,
                "end_price": end_price,
                "raw_return": raw_return,
            }
        )

    return pd.DataFrame(rows)


def compute_weekly_member_returns(
    portfolio_members: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_trading_days: int = WEEKLY_HORIZON_TRADING_DAYS,
) -> pd.DataFrame:
    """Attach raw returns over a fixed trading-day horizon after formation."""
    if horizon_trading_days < 1:
        raise ValueError("horizon_trading_days must be positive.")

    members = portfolio_members.copy()
    trading_dates = map_to_trading_dates(prices, members["formation_date"])
    price_index = pd.DatetimeIndex(prices.index).sort_values()

    rows = []
    for row in members.itertuples(index=False):
        formation_date = pd.to_datetime(row.formation_date).normalize()
        start_trade_date = trading_dates.get(formation_date)
        end_trade_date = None
        start_price = None
        end_price = None
        raw_return = None

        if start_trade_date is not None and row.ticker in prices.columns:
            start_pos = price_index.searchsorted(start_trade_date, side="left")
            end_pos = start_pos + horizon_trading_days
            if end_pos < len(price_index):
                end_trade_date = price_index[end_pos]
                start_price = prices.at[start_trade_date, row.ticker]
                end_price = prices.at[end_trade_date, row.ticker]
                if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                    raw_return = end_price / start_price - 1.0

        rows.append(
            {
                "calendar_quarter": row.calendar_quarter,
                "ticker": row.ticker,
                "side": row.side,
                "formation_date": formation_date,
                "next_formation_date": getattr(row, "next_formation_date", pd.NaT),
                "start_trade_date": start_trade_date,
                "end_trade_date": end_trade_date,
                "horizon_trading_days": horizon_trading_days,
                SIGNAL_COLUMN: getattr(row, SIGNAL_COLUMN),
                "leg_weight": row.leg_weight,
                "signed_weight": row.signed_weight,
                "start_price": start_price,
                "end_price": end_price,
                "raw_return": raw_return,
            }
        )

    return pd.DataFrame(rows)


def compute_strategy_returns(member_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate member returns into quarterly long, short, and long-short returns."""
    valid_returns = member_returns.dropna(subset=["raw_return"]).copy()
    leg_returns = (
        valid_returns.groupby(["calendar_quarter", "side"], as_index=False)
        .agg(
            leg_return=("raw_return", "mean"),
            usable_names=("ticker", "count"),
        )
        .pivot(index="calendar_quarter", columns="side")
    )
    leg_returns.columns = ["_".join(col).strip() for col in leg_returns.columns]
    leg_returns = leg_returns.reset_index()

    quarter_dates = (
        member_returns.groupby("calendar_quarter", as_index=False)
        .agg(
            formation_date=("formation_date", "first"),
            next_formation_date=("next_formation_date", "first"),
            start_trade_date=("start_trade_date", "first"),
            end_trade_date=("end_trade_date", "first"),
            selected_names=("ticker", "count"),
            priced_names=("raw_return", lambda values: values.notna().sum()),
        )
    )

    strategy_returns = quarter_dates.merge(leg_returns, on="calendar_quarter", how="left")
    strategy_returns["long_short_return"] = (
        strategy_returns["leg_return_long"] - strategy_returns["leg_return_short"]
    )
    strategy_returns["price_coverage"] = (
        strategy_returns["priced_names"] / strategy_returns["selected_names"]
    )
    return strategy_returns.sort_values("calendar_quarter")


def compute_performance(
    strategy_returns: pd.DataFrame,
    periods_per_year: float,
    period_label: str,
) -> pd.Series:
    """Compute annualized return, volatility, and Sharpe from periodic returns."""
    returns = strategy_returns["long_short_return"].dropna()
    if returns.empty:
        return pd.Series(dtype="float64")

    mean_period_return = returns.mean()
    period_volatility = returns.std(ddof=1)
    annualized_return = (1.0 + mean_period_return) ** periods_per_year - 1.0
    annualized_volatility = period_volatility * (periods_per_year ** 0.5)
    sharpe_ratio = None
    if pd.notna(period_volatility) and period_volatility != 0:
        sharpe_ratio = (periods_per_year ** 0.5) * mean_period_return / period_volatility

    return pd.Series(
        {
            "period_label": period_label,
            "periods": len(returns),
            "periods_per_year": periods_per_year,
            "mean_period_return": mean_period_return,
            "period_volatility": period_volatility,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
        }
    )


def write_outputs(
    output_dir: Path,
    formation_dates: pd.DataFrame,
    portfolio_members: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    quarterly_member_returns: pd.DataFrame,
    quarterly_strategy_returns: pd.DataFrame,
    quarterly_performance: pd.Series,
    weekly_member_returns: pd.DataFrame,
    weekly_strategy_returns: pd.DataFrame,
    weekly_performance: pd.Series,
    missing_tickers: list[str],
) -> None:
    """Persist detailed analysis artifacts for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    formation_dates.to_csv(output_dir / "formation_dates.csv", index=False)
    portfolio_members.to_csv(output_dir / "portfolio_members.csv", index=False)
    portfolio_summary.to_csv(output_dir / "portfolio_summary.csv", index=False)

    quarterly_member_returns.to_csv(output_dir / "member_returns_quarterly.csv", index=False)
    quarterly_strategy_returns.to_csv(output_dir / "strategy_returns_quarterly.csv", index=False)
    quarterly_performance.to_frame("value").to_csv(output_dir / "performance_quarterly.csv")

    weekly_member_returns.to_csv(output_dir / "member_returns_weekly.csv", index=False)
    weekly_strategy_returns.to_csv(output_dir / "strategy_returns_weekly.csv", index=False)
    weekly_performance.to_frame("value").to_csv(output_dir / "performance_weekly.csv")

    # Backward-compatible aliases for the original quarterly run.
    quarterly_member_returns.to_csv(output_dir / "member_returns.csv", index=False)
    quarterly_strategy_returns.to_csv(output_dir / "strategy_returns.csv", index=False)
    quarterly_performance.to_frame("value").to_csv(output_dir / "performance.csv")

    pd.DataFrame({"ticker": missing_tickers}).to_csv(
        output_dir / "missing_price_tickers.csv",
        index=False,
    )


def print_run_summary(
    signal_panel: pd.DataFrame,
    formation_dates: pd.DataFrame,
    portfolio_members: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    quarterly_strategy_returns: pd.DataFrame,
    quarterly_performance: pd.Series,
    weekly_strategy_returns: pd.DataFrame,
    weekly_performance: pd.Series,
    missing_tickers: list[str],
    output_dir: Path,
) -> None:
    """Print high-signal run diagnostics without dumping all holdings."""
    print(f"Signal: {SIGNAL_COLUMN}")
    print(f"Signal panel rows: {len(signal_panel):,}")
    print(f"Unique tickers: {signal_panel['ticker'].nunique():,}")
    print(f"Formation quarters: {len(formation_dates):,}")
    print(f"Portfolio member rows: {len(portfolio_members):,}")

    print("\nFormation date range:")
    print(
        formation_dates[["calendar_quarter", "formation_date", "next_formation_date", "tickers"]]
        .head(3)
        .to_string(index=False)
    )
    print("...")
    print(
        formation_dates[["calendar_quarter", "formation_date", "next_formation_date", "tickers"]]
        .tail(3)
        .to_string(index=False)
    )

    leg_size_summary = (
        portfolio_summary.groupby("side")["names"]
        .agg(["min", "median", "max"])
        .rename(columns={"min": "min_names", "median": "median_names", "max": "max_names"})
    )
    print(f"\nPortfolio sorting: top/bottom {LEG_FRACTION:.0%}")
    print(leg_size_summary.to_string())

    print("\nQuarterly price data coverage:")
    print(
        quarterly_strategy_returns[["calendar_quarter", "selected_names", "priced_names", "price_coverage"]]
        .to_string(index=False)
    )
    if missing_tickers:
        print(f"Missing price tickers ({len(missing_tickers)}): {', '.join(missing_tickers)}")

    print("\nQuarterly strategy returns:")
    print(
        quarterly_strategy_returns[
            [
                "calendar_quarter",
                "start_trade_date",
                "end_trade_date",
                "leg_return_long",
                "leg_return_short",
                "long_short_return",
            ]
        ].to_string(index=False)
    )

    print(f"\nWeekly strategy returns ({WEEKLY_HORIZON_TRADING_DAYS} trading days):")
    print(
        weekly_strategy_returns[
            [
                "calendar_quarter",
                "start_trade_date",
                "end_trade_date",
                "leg_return_long",
                "leg_return_short",
                "long_short_return",
            ]
        ].to_string(index=False)
    )

    print("\nAnnualized quarterly-horizon performance:")
    print(quarterly_performance.to_string())
    print("\nAnnualized weekly-horizon performance:")
    print(weekly_performance.to_string())
    print(f"\nSaved outputs to: {output_dir}")
