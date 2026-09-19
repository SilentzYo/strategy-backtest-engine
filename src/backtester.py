import pandas as pd

def run_backtest(prices, signals, cost=0.001):
    df = pd.DataFrame(index=prices.index)
    df["Close"] = prices["Close"]
    df["signal"] = signals
    df["position"] = df["signal"].shift(1)

    #Daily Returns
    df["returns"] = df["Close"].pct_change()

    #Strategy Returns
    df["strategy_returns"] = df["position"] * df["returns"]

    #Transaction Costs
    df["trade"] = df["position"].diff().abs()
    df["strategy_returns"] -= df["trade"].fillna(0) * cost

    #Equity Curve
    df["equity_curve"] = (1 + df["strategy_returns"].fillna(0)).cumprod()

    return df

def buy_and_hold(prices):
    returns = prices["Close"].pct_change()
    equity_curve = (1 + returns.fillna(0)).cumprod()
    return equity_curve