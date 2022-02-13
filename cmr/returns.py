import logging

import cvxportfolio as cvx
import pandas as pd
import sklearn.linear_model as linear_model
import sklearn.preprocessing as pre
from lazy import lazy
from sklearn.pipeline import make_pipeline

from .data_loader import load_features


class TaReturnsForecast(cvx.ReturnsForecast):

    def __init__(self, symbols: list, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """

        :param symbols: universe
        :param start: start timestamp
        :param end: end timestamp
        """
        self.symbols = symbols
        self.start = start
        self.end = end
        super().__init__(self.get_value())

    @lazy
    def alpha_source(self):
        return load_features(self.symbols, self.start, self.end)

    @lazy
    def test_start(self):
        return ((self.end - self.start) * .9 + self.start).round('D')

    @lazy
    def test_end(self):
        return ((self.end - self.start) * .95 + self.start).round('D')

    @lazy
    def model(self):
        return make_pipeline(
            pre.RobustScaler(with_centering=False),
            linear_model.LinearRegression()
        )

    def get_value(self):
        """
        Get signal value
        """
        logging.info("built signals from %s to %s" % (self.start, self.end))

        # shift y
        af = self.alpha_source
        af['ret'] = af.groupby('symbol')['ret'].shift(-1).fillna(0)

        # split train and test
        x = self.alpha_source.drop(columns=['ret'])
        y = self.alpha_source['ret'] - self.alpha_source.ret.groupby(level=0).transform(lambda x_: x_.mean())
        x_train = x.loc[x.index.get_level_values(0) < self.test_start]
        y_train = y.loc[y.index.get_level_values(0) < self.test_start]
        x_test = x.loc[(x.index.get_level_values(0) >= self.test_start) & (x.index.get_level_values(0) < self.test_end)]

        self.model.fit(x_train, y_train)
        predictions = pd.Series(index=x_test.index, data=self.model.predict(x_test)).unstack().shift(1).fillna(0)

        # construct outputs
        predictions['cash'] = 0
        return predictions  # FIXME: fill zero is too brute-force
