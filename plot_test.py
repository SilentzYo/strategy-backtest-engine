from src.data_loader import load_prices
from src.strategies.moving_average import MovingAverageStrategy
from src.backtester import run_backtest, buy_and_hold
from src.plotting import plot_equity_curves, plot_drawdown

strategy = MovingAverageStrategy(short_window=20, long_window=50)

for ticker in ["AAPL", "AMD"]:
    prices = load_prices(ticker)
    signals = strategy.generate_signals(prices)
    results = run_backtest(prices, signals)

    bh_equity = buy_and_hold(prices)

    plot_equity_curves(
        {
            "MA Crossover": results["equity_curve"],
            "Buy & Hold": bh_equity,
        },
        title=f"{ticker}: Strategy vs Buy & Hold",
        save_path=f"results/figures/{ticker}_equity_curve.png",
    )

    plot_drawdown(
        results["equity_curve"],
        title=f"{ticker}: MA Crossover Drawdown",
        save_path=f"results/figures/{ticker}_drawdown.png",
    )