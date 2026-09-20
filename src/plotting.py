import matplotlib.pyplot as plt


def plot_equity_curves(curves, title, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    for label, curve in curves.items():
        ax.plot(curve.index, curve.values, label=label, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def compute_drawdown_series(equity_curve):
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def plot_drawdown(equity_curve, title, save_path=None):
    drawdown = compute_drawdown_series(equity_curve)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="darkred", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()