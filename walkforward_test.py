from src.data_loader import load_prices
from src.optimization import walk_forward_optimize

TICKER = "AAPL"

prices = load_prices(TICKER)

short_range = range(5, 51, 5)
long_range = range(50, 201, 10)

print("Running walk-forward optimization (this will take a bit)...")
wf_results = walk_forward_optimize(
    prices,
    short_range,
    long_range,
    train_years=3,
    test_months=6,
)

print("\nWalk-forward windows:")
print(wf_results[["train_end", "test_end", "short_window", "long_window", "train_sharpe", "test_sharpe"]])

avg_test_sharpe = wf_results["test_sharpe"].mean()
print(f"\nAverage out-of-sample Sharpe across all windows: {avg_test_sharpe:.2f}")

wf_results.to_csv("results/reports/walkforward_results.csv", index=False)