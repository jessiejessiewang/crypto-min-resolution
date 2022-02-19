import logging

import cvxportfolio as cvx
import pandas as pd
import sklearn.impute as impute
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import sklearn.preprocessing as pre
from lazy import lazy
from sklearn.pipeline import make_pipeline

from .data_loader import load_features


class TaReturnsForecast(cvx.ReturnsForecast):

    def __init__(self, symbols: list, start: pd.Timestamp, end: pd.Timestamp, resample_rule: str = '1H') -> None:
        """

        :param symbols: universe
        :param start: start timestamp
        :param end: end timestamp
        :param resample_rule：frequency
        """

        self.symbols = symbols
        self.start = start
        self.end = end
        self.resample_rule = resample_rule
        self.fit_model()
        super().__init__(self.get_prediction())

    @lazy
    def alpha_source(self):
        return load_features(self.symbols, self.start, self.end, self.resample_rule)

    @lazy
    def test_start(self):
        return ((self.end - self.start) * .9 + self.start).round('D')

    @lazy
    def test_end(self):
        return ((self.end - self.start) * .95 + self.start).round('D')

    @lazy
    def model(self):
        return make_pipeline(
            impute.SimpleImputer(),
            pre.RobustScaler(with_centering=False),
            linear_model.LinearRegression()
        )

    def fit_model(self):
        logging.info("built signals from %s to %s" % (self.start, self.end))

        # shift y
        af = self.alpha_source.copy()
        af['ret1d'] = af.groupby('symbol')['ret'].shift(-1).fillna(0)

        # split train and test
        x = af.drop(columns=['ret1d'])
        y = af['ret1d']
        x_train = x  # x.loc[(x.index.get_level_values(0) < self.test_start)]
        y_train = y  # y.loc[(y.index.get_level_values(0) < self.test_start)]
        self.model.fit(x_train, y_train)

        # print out in-sample evaluation
        y_pred = self.model.predict(x_train)
        print("Mean absolute error (in-sample): %.6f" % metrics.mean_absolute_error(y_train, y_pred))
        print("Mean squared error (in-sample): %.6f" % metrics.mean_squared_error(y_train, y_pred))

    def get_prediction(self):
        """
        Get signal value
        """
        logging.info("built signals from %s to %s" % (self.start, self.end))

        # shift y
        af = self.alpha_source.copy()
        af['ret1d'] = af.groupby('symbol')['ret'].shift(-1).fillna(0)

        # split train and test
        x, y = af.drop(columns=['ret1d']), af['ret1d']
        x_test = x.loc[(x.index.get_level_values(0) >= self.test_start) & (x.index.get_level_values(0) < self.test_end)]
        y_true = y.loc[(y.index.get_level_values(0) >= self.test_start) & (y.index.get_level_values(0) < self.test_end)]
        y_pred = self.model.predict(x_test)

        # print out-of-sample evaluation
        print("Mean absolute error (out-of-sample): %.6f" % metrics.mean_absolute_error(y_true, y_pred))
        print("Mean squared error (out-of-sample): %.6f" % metrics.mean_squared_error(y_true, y_pred))

        # FIXME: fill zero is too brute-force
        self.returns = pd.Series(index=x_test.index, data=y_pred).unstack().shift(1).fillna(0)

        # construct outputs
        self.returns['cash'] = 0
        return self.returns
