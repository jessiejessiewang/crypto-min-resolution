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
        self.fit_model()
        super().__init__(self.get_prediction())

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

    def fit_model(self):
        logging.info("built signals from %s to %s" % (self.start, self.end))

        # shift y
        af = self.alpha_source
        af['ret1d'] = af.groupby('symbol')['ret'].shift(-1).fillna(0)

        # split train and test
        x = af.drop(columns=['ret1d'])
        y = af['ret1d'] - af.ret.groupby(level=0).transform(lambda x_: x_.mean())
        x_train = x.loc[x.index.get_level_values(0) < self.test_start]
        y_train = y.loc[y.index.get_level_values(0) < self.test_start]

        self.model.fit(x_train, y_train)

    def get_prediction(self):
        """
        Get signal value
        """
        logging.info("built signals from %s to %s" % (self.start, self.end))

        # shift y
        af = self.alpha_source
        af['ret1d'] = af.groupby('symbol')['ret'].shift(-1).fillna(0)

        # split train and test
        x = af.drop(columns=['ret1d'])
        x_test = x.loc[(x.index.get_level_values(0) >= self.test_start) & (x.index.get_level_values(0) < self.test_end)]
        # FIXME: fill zero is too brute-force
        self.returns = pd.Series(index=x_test.index, data=self.model.predict(x_test)).unstack().shift(1).fillna(0)

        # construct outputs
        self.returns['cash'] = 0
        return self.returns
