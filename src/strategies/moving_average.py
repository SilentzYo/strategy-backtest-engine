from .base import Strategy

class MovingAverageStrategy(Strategy):
    def __init__(self, short_window = 20, long_window = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, prices):
        short_ma = prices["Close"].rolling(window=self.short_window).mean()
        long_ma = prices["Close"].rolling(window=self.long_window).mean()

        return (short_ma > long_ma).astype(int)