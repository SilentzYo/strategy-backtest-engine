from src.data_loader import load_prices
from src.optimization import split_prices, grid_search_ma, evaluate_params

TICKER = "AAPL"
SPLIT_DATE = "2020-01-01"

prices = load_prices(TICKER)
in_sample, out_of_sample = split_prices(prices, SPLIT_DATE)

print(f"In-sample:     {in_sample.index.min().date()} to {in_sample.index.max().date()} ({len(in_sample)} days)")
print(f"Out-of-sample: {out_of_sample.index.min().date()} to {out_of_sample.index.max().date()} ({len(out_of_sample)} days)")

short_range = range(5, 51, 5)
long_range = range(50, 201, 10)

print("\nRunning grid search on in-sample data...")
grid_results = grid_search_ma(in_sample, short_range, long_range)

print("\nTop 5 parameter combos (in-sample):")
print(grid_results.head(5))

best = grid_results.iloc[0]
best_short, best_long = int(best["short_window"]), int(best["long_window"])
print(f"\nBest combo: short={best_short}, long={best_long}, in-sample Sharpe={best['sharpe']:.2f}")

oos_sharpe = evaluate_params(out_of_sample, best_short, best_long)
print(f"Same combo, OUT-OF-SAMPLE Sharpe: {oos_sharpe:.2f}")

print(f"\nDegradation: {best['sharpe']:.2f} -> {oos_sharpe:.2f}")