import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate/periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)

def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def total_return(equity_curve):
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def win_rate(returns):
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).sum() / len(active)


def summarize(results):
    sr = sharpe_ratio(results["strategy_returns"].dropna())
    mdd = max_drawdown(results["equity_curve"])
    tr = total_return(results["equity_curve"])
    wr = win_rate(results["strategy_returns"])

    print(f"Total Return:   {tr:.2%}")
    print(f"Sharpe Ratio:   {sr:.2f}")
    print(f"Max Drawdown:   {mdd:.2%}")
    print(f"Win Rate:       {wr:.2%}")

    return {"total_return": tr, "sharpe": sr, "max_drawdown": mdd, "win_rate": wr}