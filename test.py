import pandas as pd
from src.data_loader import load_prices
from src.strategies.moving_average import MovingAverageStrategy
from src.backtester import run_backtest, buy_and_hold
from src.metrics import sharpe_ratio, max_drawdown, total_return, win_rate

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]

strategy = MovingAverageStrategy(short_window=20, long_window=50)

rows = []
for ticker in TICKERS:
    prices = load_prices(ticker)
    signals = strategy.generate_signals(prices)
    results = run_backtest(prices, signals)

    strat_returns = results["strategy_returns"].dropna()
    bh_equity = buy_and_hold(prices)

    rows.append({
        "Ticker": ticker,
        "Strategy Return": total_return(results["equity_curve"]),
        "Buy & Hold Return": bh_equity.iloc[-1] - 1,
        "Sharpe": sharpe_ratio(strat_returns),
        "Max Drawdown": max_drawdown(results["equity_curve"]),
        "Win Rate": win_rate(strat_returns),
    })

comparison = pd.DataFrame(rows).set_index("Ticker")

pd.set_option("display.float_format", lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}")
print(comparison)

comparison.to_csv("results/reports/ma_crossover_comparison.csv")