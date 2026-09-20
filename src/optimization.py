import itertools
import pandas as pd

from src.strategies.moving_average import MovingAverageStrategy
from src.backtester import run_backtest
from src.metrics import sharpe_ratio


def split_prices(prices, split_date):
    in_sample = prices[prices.index < split_date].copy()
    out_of_sample = prices[prices.index >= split_date].copy()
    return in_sample, out_of_sample


def grid_search_ma(prices, short_range, long_range, cost=0.001):
    results = []

    for short_window, long_window in itertools.product(short_range, long_range):
        if short_window >= long_window:
            continue

        strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
        signals = strategy.generate_signals(prices)
        backtest = run_backtest(prices, signals, cost=cost)

        sr = sharpe_ratio(backtest["strategy_returns"].dropna())

        results.append({
            "short_window": short_window,
            "long_window": long_window,
            "sharpe": sr,
        })

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df


def evaluate_params(prices, short_window, long_window, cost=0.001):
    strategy = MovingAverageStrategy(short_window=short_window, long_window=long_window)
    signals = strategy.generate_signals(prices)
    backtest = run_backtest(prices, signals, cost=cost)
    return sharpe_ratio(backtest["strategy_returns"].dropna())

def walk_forward_optimize(prices, short_range, long_range, train_years=3, test_months=6, transaction_cost=0.001):
    results = []

    start_date = prices.index.min()
    end_date = prices.index.max()

    train_delta = pd.DateOffset(years=train_years)
    test_delta = pd.DateOffset(months=test_months)

    window_start = start_date
    train_end = window_start + train_delta
    test_end = train_end + test_delta

    while test_end <= end_date:
        train_data = prices[(prices.index >= window_start) & (prices.index < train_end)]
        test_data = prices[(prices.index >= train_end) & (prices.index < test_end)]

        if len(train_data) < 100 or len(test_data) < 20:
            break

        grid = grid_search_ma(train_data, short_range, long_range, transaction_cost)
        if grid.empty:
            break

        best = grid.iloc[0]
        short_w, long_w = int(best["short_window"]), int(best["long_window"])

        test_sharpe = evaluate_params(test_data, short_w, long_w, transaction_cost)

        results.append({
            "train_start": window_start.date(),
            "train_end": train_end.date(),
            "test_end": test_end.date(),
            "short_window": short_w,
            "long_window": long_w,
            "train_sharpe": best["sharpe"],
            "test_sharpe": test_sharpe,
        })

        window_start += test_delta
        train_end += test_delta
        test_end += test_delta

    return pd.DataFrame(results)