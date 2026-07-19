from __future__ import annotations

import contextlib
import io

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - cosmetic fallback
    tqdm = None

from src.data_analysis.data_fetcher.data_fetcher_class import DataFetcher
from src.data_analysis.sharpe_ratio.pipeline import (
    OUTPUT_DIR,
    PRICE_CACHE_FILE,
    build_formation_dates,
    build_portfolio_members,
    build_signal_panel,
    compute_performance,
    compute_strategy_returns,
    compute_weekly_member_returns,
    load_cached_adjusted_closes,
)
from src.data_collection.consts import DB_PARAMS


HORIZON_GRID = range(1, 64)
LEG_FRACTION_GRID = [fraction / 100 for fraction in range(10, 45, 5)]
TRADING_DAYS_PER_YEAR = 252
OPTIMIZATION_OUTPUT_FILE = OUTPUT_DIR / "horizon_optimization.csv"


def load_signal_panel() -> pd.DataFrame:
    """Fetch the sentiment signal universe for the optimization run."""
    with contextlib.redirect_stdout(io.StringIO()):
        fetcher = DataFetcher(DB_PARAMS)
        return build_signal_panel(fetcher)


def build_optimization_base(
    signal_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build the common formation-date backbone used by every grid point."""
    return build_formation_dates(signal_panel)


def load_optimization_prices(
    signal_panel: pd.DataFrame,
    formation_dates: pd.DataFrame,
    max_leg_fraction: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the existing price cache without downloading new prices."""
    widest_members = build_portfolio_members(
        signal_panel,
        formation_dates,
        leg_fraction=max_leg_fraction,
    )
    tickers = sorted(widest_members["ticker"].unique())
    prices = load_cached_adjusted_closes(PRICE_CACHE_FILE)
    if prices.empty:
        raise RuntimeError(
            f"No cached prices found at {PRICE_CACHE_FILE}. "
            "Run src.data_analysis.sharpe_ratio.main first to build the cache."
        )

    print(
        f"Using cached prices from {PRICE_CACHE_FILE} "
        f"({prices.shape[0]:,} dates, {prices.shape[1]:,} tickers; "
        f"{prices.index.min().date()} to {prices.index.max().date()})"
    )
    returned_tickers = set(prices.columns[prices.notna().any(axis=0)])
    missing_tickers = sorted(set(tickers) - returned_tickers)
    return prices, missing_tickers


def evaluate_grid_point(
    signal_panel: pd.DataFrame,
    formation_dates: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_trading_days: int,
    leg_fraction: float,
) -> dict[str, float | int | str]:
    """Evaluate one horizon/fraction pair and return one optimization row."""
    portfolio_members = build_portfolio_members(
        signal_panel,
        formation_dates,
        leg_fraction=leg_fraction,
    )
    member_returns = compute_weekly_member_returns(
        portfolio_members,
        prices,
        horizon_trading_days=horizon_trading_days,
    )
    strategy_returns = compute_strategy_returns(member_returns)
    periods_per_year = TRADING_DAYS_PER_YEAR / horizon_trading_days
    performance = compute_performance(
        strategy_returns,
        periods_per_year=periods_per_year,
        period_label=f"{horizon_trading_days}_trading_days",
    )

    row = {
        "horizon_trading_days": horizon_trading_days,
        "leg_fraction": leg_fraction,
        "long_short_fraction": 2 * leg_fraction,
        "avg_price_coverage": strategy_returns["price_coverage"].mean(),
        "min_price_coverage": strategy_returns["price_coverage"].min(),
        "selected_names_mean": strategy_returns["selected_names"].mean(),
        "priced_names_mean": strategy_returns["priced_names"].mean(),
    }
    row.update(performance.to_dict())
    return row


def iter_optimization_grid() -> list[tuple[float, int]]:
    """Return all leg-fraction/horizon grid points in run order."""
    return [
        (leg_fraction, horizon_trading_days)
        for leg_fraction in LEG_FRACTION_GRID
        for horizon_trading_days in HORIZON_GRID
    ]


def with_progress(
    grid_points: list[tuple[float, int]],
):
    """Wrap grid points with a progress bar when tqdm is available."""
    if tqdm is not None:
        return tqdm(
            grid_points,
            total=len(grid_points),
            desc="Optimizing horizon/fraction",
            unit="grid point",
        )

    print(f"Optimizing {len(grid_points)} horizon/fraction grid points...")
    return grid_points


def run_optimization_grid(
    signal_panel: pd.DataFrame,
    formation_dates: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Run the horizon/fraction double grid."""
    rows = []
    for leg_fraction, horizon_trading_days in with_progress(iter_optimization_grid()):
        rows.append(
            evaluate_grid_point(
                signal_panel=signal_panel,
                formation_dates=formation_dates,
                prices=prices,
                horizon_trading_days=horizon_trading_days,
                leg_fraction=leg_fraction,
            )
        )

    results = pd.DataFrame(rows)
    return results.sort_values(["leg_fraction", "horizon_trading_days"])


def save_optimization_results(results: pd.DataFrame) -> None:
    """Persist the full grid so we can inspect more than only the optimum."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OPTIMIZATION_OUTPUT_FILE, index=False)


def print_optimization_summary(
    results: pd.DataFrame,
    missing_tickers: list[str],
) -> None:
    """Print the best grid point and basic diagnostics."""
    valid_results = results.dropna(subset=["sharpe_ratio"])
    if valid_results.empty:
        print("No valid Sharpe ratios produced by the optimization grid.")
        print(f"Saved optimization grid to: {OPTIMIZATION_OUTPUT_FILE}")
        return

    best = valid_results.sort_values("sharpe_ratio", ascending=False).iloc[0]
    print("Best horizon/fraction grid point:")
    print(
        best[
            [
                "horizon_trading_days",
                "leg_fraction",
                "long_short_fraction",
                "periods",
                "periods_per_year",
                "mean_period_return",
                "period_volatility",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "avg_price_coverage",
                "min_price_coverage",
            ]
        ].to_string()
    )

    print("\nTop 10 grid points by Sharpe:")
    print(
        valid_results.sort_values("sharpe_ratio", ascending=False)
        .head(10)[
            [
                "horizon_trading_days",
                "leg_fraction",
                "sharpe_ratio",
                "annualized_return",
                "annualized_volatility",
                "avg_price_coverage",
            ]
        ]
        .to_string(index=False)
    )

    if missing_tickers:
        print(f"\nMissing price tickers ({len(missing_tickers)}): {', '.join(missing_tickers)}")
    print(f"\nSaved optimization grid to: {OPTIMIZATION_OUTPUT_FILE}")


def run_optimization_pipeline() -> None:
    """Run the simple double-grid Sharpe optimization."""
    signal_panel = load_signal_panel()
    formation_dates = build_optimization_base(signal_panel)
    prices, missing_tickers = load_optimization_prices(
        signal_panel=signal_panel,
        formation_dates=formation_dates,
        max_leg_fraction=max(LEG_FRACTION_GRID),
    )
    results = run_optimization_grid(
        signal_panel=signal_panel,
        formation_dates=formation_dates,
        prices=prices,
    )
    save_optimization_results(results)
    print_optimization_summary(results, missing_tickers)


def main() -> None:
    run_optimization_pipeline()


if __name__ == "__main__":
    main()
