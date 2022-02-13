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

        :param symbols:
        :param start:
        :param end:
        """
        self.symbols = symbols
        self.start = start
        self.end = end
        super().__init__(self.get_value())

    @lazy
    def alpha_source(self):
        return load_features(self.symbols, self.start, self.end)

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

        x = self.alpha_source.drop(columns=['ret_fwd1'])
        y = self.alpha_source['ret_fwd1'] - self.alpha_source.ret_fwd1.groupby(level=0).transform(lambda x_: x_.mean())
        self.model.fit(x, y)
        predictions = pd.Series(index=x.index, data=self.model.predict(x)).unstack().shift(1).fillna(0)
        predictions['cash'] = 0
        return predictions  # FIXME: fill zero is too brute-force
