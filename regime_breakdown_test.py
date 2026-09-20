import pandas as pd

from src.data_loader import load_prices
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.backtester import run_backtest, buy_and_hold
from src.metrics import sharpe_ratio, total_return

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]

REGIMES = {
    "2015-2016 (choppy)": ("2015-01-01", "2017-01-01"),
    "2017-2019 (calm bull)": ("2017-01-01", "2020-01-01"),
    "2020 (COVID crash/recovery)": ("2020-01-01", "2021-01-01"),
    "2021 (melt-up)": ("2021-01-01", "2022-01-01"),
    "2022 (rate-hike selloff)": ("2022-01-01", "2023-01-01"),
    "2023 (AI rally)": ("2023-01-01", "2024-01-01"),
}

strategies = {
    "MA Crossover": MovingAverageStrategy(short_window=20, long_window=50),
    "Mean Reversion": MeanReversionStrategy(window=20, num_std=2),
}

rows = []

for ticker in TICKERS:
    prices = load_prices(ticker)

    for regime_name, (start, end) in REGIMES.items():
        regime_prices = prices[(prices.index >= start) & (prices.index < end)]

        if len(regime_prices) < 30:
            continue

        bh_equity = buy_and_hold(regime_prices)
        bh_return = bh_equity.iloc[-1] - 1

        for strat_name, strategy in strategies.items():
            signals = strategy.generate_signals(regime_prices)
            results = run_backtest(regime_prices, signals)
            strat_returns = results["strategy_returns"].dropna()

            rows.append({
                "Ticker": ticker,
                "Regime": regime_name,
                "Strategy": strat_name,
                "Return": total_return(results["equity_curve"]),
                "Sharpe": sharpe_ratio(strat_returns),
                "Buy & Hold Return": bh_return,
            })

breakdown = pd.DataFrame(rows).set_index(["Regime", "Ticker", "Strategy"])
pd.set_option("display.float_format", lambda x: f"{x:.1%}" if pd.notna(x) and abs(x) < 20 else f"{x:.2f}")

print(breakdown)
breakdown.to_csv("results/reports/regime_breakdown.csv")