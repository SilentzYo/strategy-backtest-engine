import os
import pandas as pd
import yfinance as yf

raw_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def load_data(tickers, start, end, save=True):
    data = yf.download(tickers, start=start, end=end, group_by="ticker")

    result = {} 
    for ticker in tickers:
        if len(tickers) > 0:
            df = data[ticker].copy()
        else:
            df = data.copy()

        df = df.dropna(how="all")
        result[ticker] = df

        if save:
            os.makedirs(raw_data_dir, exist_ok=True)
            path = os.path.join(raw_data_dir, f"{ticker}.csv")
            df.to_csv(path)

    return result

def load_prices(ticker):
    path = os.path.join(raw_data_dir, f"{ticker}.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)

if __name__ == "__main__":
    tickers = ["SPY", "AAPL", "MSFT", "NVDA", "AMD"]
    data = load_data(tickers, start="2015-01-01", end="2024-01-01")

    for ticker, df in data.items():
        print(f"{ticker}: {df.shape[0]} rows, {df.index.min().date()} to{df.index.max().date()}")
