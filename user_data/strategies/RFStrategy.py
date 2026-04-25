import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy


class RFStrategy(IStrategy):
    """
    Random Forest FreqAI strategy for BTC/USDT and ETH/USDT
    Uses technical indicators as features to predict price direction
    """

    # Strategy settings
    minimal_roi = {"0": 0.03}  # Exit at 3% profit
    stoploss = -0.02  # Exit at 2% loss
    timeframe = "5m"
    can_short = False
    use_exit_signal = True
    freqai_info = {}

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Define all the features (inputs) the ML model will train on.
        More good features = better predictions.
        """
        # Price momentum features
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-period_{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-period_{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-cci-period_{period}"] = ta.CCI(dataframe, timeperiod=period)

        # Moving average distance (how far price is from MA)
        dataframe[f"%-close-mean-period_{period}"] = (
            dataframe["close"] / dataframe["close"].rolling(period).mean() - 1
        )

        # Volume features
        dataframe[f"%-volume-mean-period_{period}"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean() - 1
        )

        # Bollinger Band position
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        dataframe[f"%-bb_width-period_{period}"] = (
            bollinger["upper"] - bollinger["lower"]
        ) / bollinger["mid"]
        dataframe[f"%-bb_position-period_{period}"] = (dataframe["close"] - bollinger["lower"]) / (
            bollinger["upper"] - bollinger["lower"]
        )

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Standard features computed once (not per period).
        """
        # Raw OHLCV ratios
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        # Price change features
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Define what the model is trying to predict.
        Here: will price be higher in 12 candles (1 hour on 5m timeframe)?
        """
        dataframe["&-price_up"] = (dataframe["close"].shift(-12) > dataframe["close"]).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["&-price_up_predicted"] == 1) & (dataframe["do_predict"] == 1), "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["&-price_up_predicted"] == 0) & (dataframe["do_predict"] == 1), "exit_long"
        ] = 1
        return dataframe
