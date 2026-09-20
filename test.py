import pandas as pd
from src.data_loader import load_prices
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.backtester import run_backtest, buy_and_hold
from src.metrics import sharpe_ratio, max_drawdown, total_return, win_rate

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]

strategies = {
    "MA Crossover": MovingAverageStrategy(short_window=20, long_window=50),
    "Mean Reversion": MeanReversionStrategy(window=20, num_std=2),
}

rows = []
for ticker in TICKERS:
    prices = load_prices(ticker)
    bh_equity = buy_and_hold(prices)

    for strat_name, strategy in strategies.items():
        signals = strategy.generate_signals(prices)
        results = run_backtest(prices, signals)
        strat_returns = results["strategy_returns"].dropna()

        rows.append({
            "Ticker": ticker,
            "Strategy": strat_name,
            "Return": total_return(results["equity_curve"]),
            "Sharpe": sharpe_ratio(strat_returns),
            "Max Drawdown": max_drawdown(results["equity_curve"]),
            "Win Rate": win_rate(strat_returns),
        })

    rows.append({
        "Ticker": ticker,
        "Strategy": "Buy & Hold",
        "Return": bh_equity.iloc[-1] - 1,
        "Sharpe": None,
        "Max Drawdown": max_drawdown(bh_equity),
        "Win Rate": None,
    })

comparison = pd.DataFrame(rows).set_index(["Ticker", "Strategy"])
pd.set_option("display.float_format", lambda x: f"{x:.2%}" if pd.notna(x) and abs(x) < 10 else str(x))
print(comparison)

comparison.to_csv("results/reports/strategy_comparison.csv")