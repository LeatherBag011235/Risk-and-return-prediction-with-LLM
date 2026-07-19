from __future__ import annotations

import contextlib
import io

import pandas as pd

from src.data_analysis.data_fetcher.data_fetcher_class import DataFetcher
from src.data_analysis.sharpe_ratio.pipeline import (
    OUTPUT_DIR,
    WEEKLY_HORIZON_TRADING_DAYS,
    build_formation_dates,
    build_portfolio_members,
    build_portfolio_summary,
    build_signal_panel,
    compute_member_returns,
    compute_performance,
    compute_strategy_returns,
    compute_weekly_member_returns,
    download_adjusted_closes,
    print_run_summary,
    write_outputs,
)
from src.data_collection.consts import DB_PARAMS


def load_signal_panel() -> pd.DataFrame:
    """Fetch the sentiment signal universe for the current pipeline run."""
    with contextlib.redirect_stdout(io.StringIO()):
        fetcher = DataFetcher(DB_PARAMS)
        return build_signal_panel(fetcher)


def build_current_portfolio(
    signal_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build formation dates, top/bottom sorted members, and a compact summary."""
    formation_dates = build_formation_dates(signal_panel)
    portfolio_members = build_portfolio_members(signal_panel, formation_dates)
    portfolio_summary = build_portfolio_summary(portfolio_members)
    return formation_dates, portfolio_members, portfolio_summary


def load_current_prices(
    portfolio_members: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Load cached prices, downloading only when the cache cannot cover this run."""
    priced_members = portfolio_members.dropna(subset=["next_formation_date"]).copy()
    min_formation_date = portfolio_members["formation_date"].min()
    max_price_date = max(
        priced_members["next_formation_date"].max(),
        portfolio_members["formation_date"].max()
        + pd.offsets.BDay(WEEKLY_HORIZON_TRADING_DAYS),
    )
    tickers = sorted(portfolio_members["ticker"].unique())
    prices = download_adjusted_closes(
        tickers=tickers,
        start_date=min_formation_date,
        end_date=max_price_date,
    )

    returned_tickers = set(prices.columns[prices.notna().any(axis=0)])
    missing_tickers = sorted(set(tickers) - returned_tickers)
    return prices, missing_tickers


def compute_current_returns(
    portfolio_members: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute quarterly and seven-trading-day strategy returns."""
    quarterly_member_returns = compute_member_returns(portfolio_members, prices)
    quarterly_strategy_returns = compute_strategy_returns(quarterly_member_returns)
    quarterly_performance = compute_performance(
        quarterly_strategy_returns,
        periods_per_year=4,
        period_label="quarter",
    )

    weekly_member_returns = compute_weekly_member_returns(portfolio_members, prices)
    weekly_strategy_returns = compute_strategy_returns(weekly_member_returns)
    weekly_performance = compute_performance(
        weekly_strategy_returns,
        periods_per_year=52,
        period_label="week",
    )
    return (
        quarterly_member_returns,
        quarterly_strategy_returns,
        quarterly_performance,
        weekly_member_returns,
        weekly_strategy_returns,
        weekly_performance,
    )


def run_current_pipeline() -> None:
    """Run the current fixed quarterly and seven-trading-day Sharpe pipeline."""
    signal_panel = load_signal_panel()
    formation_dates, portfolio_members, portfolio_summary = build_current_portfolio(
        signal_panel
    )
    prices, missing_tickers = load_current_prices(portfolio_members)
    (
        quarterly_member_returns,
        quarterly_strategy_returns,
        quarterly_performance,
        weekly_member_returns,
        weekly_strategy_returns,
        weekly_performance,
    ) = compute_current_returns(portfolio_members, prices)

    write_outputs(
        output_dir=OUTPUT_DIR,
        formation_dates=formation_dates,
        portfolio_members=portfolio_members,
        portfolio_summary=portfolio_summary,
        quarterly_member_returns=quarterly_member_returns,
        quarterly_strategy_returns=quarterly_strategy_returns,
        quarterly_performance=quarterly_performance,
        weekly_member_returns=weekly_member_returns,
        weekly_strategy_returns=weekly_strategy_returns,
        weekly_performance=weekly_performance,
        missing_tickers=missing_tickers,
    )
    print_run_summary(
        signal_panel=signal_panel,
        formation_dates=formation_dates,
        portfolio_members=portfolio_members,
        portfolio_summary=portfolio_summary,
        quarterly_strategy_returns=quarterly_strategy_returns,
        quarterly_performance=quarterly_performance,
        weekly_strategy_returns=weekly_strategy_returns,
        weekly_performance=weekly_performance,
        missing_tickers=missing_tickers,
        output_dir=OUTPUT_DIR,
    )


def main() -> None:
    run_current_pipeline()


if __name__ == "__main__":
    main()
