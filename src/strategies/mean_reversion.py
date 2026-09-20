from .base import Strategy


class MeanReversionStrategy(Strategy):
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, prices):
        close = prices["Close"]

        rolling_mean = close.rolling(window=self.window).mean()
        rolling_std = close.rolling(window=self.window).std()

        lower_band = rolling_mean - self.num_std * rolling_std

        oversold = close < lower_band

        signal = self._build_positions(close, rolling_mean, oversold)

        return signal

    def _build_positions(self, close, rolling_mean, oversold):
        position = 0
        signals = []

        for i in range(len(close)):
            if position == 0 and oversold.iloc[i]:
                position = 1
            elif position == 1 and close.iloc[i] >= rolling_mean.iloc[i]:
                position = 0

            signals.append(position)

        return signals