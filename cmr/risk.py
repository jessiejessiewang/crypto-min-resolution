import logging

import pandas as pd

from .data_loader import load_cov


class ReturnsCovRiskModel(object):

    def __init__(self, symbols: list, start: pd.Timestamp, end: pd.Timestamp, resample_rule: str = '1H') -> None:
        """

        :param symbols: list of universe
        :param start: start date
        :param end: end date
        :param resample_rule: frequency
        """
        self.symbols = symbols
        self.start = start
        self.end = end
        self.resample_rule = resample_rule

    def get_value(self):
        """
        Get risk model
        """
        logging.info("built risk model from %s to %s" % (self.start, self.end))
        return_cov = load_cov(self.symbols, self.start, self.end, 180, self.resample_rule).droplevel(1)
        return return_cov  # Barra risk model in equities in practice
