import cvxportfolio as cp
import pandas as pd

from .returns import TaReturnsForecast
from .risk import ReturnsCovRiskModel


class CryptoStatArb(cp.SinglePeriodOpt):

    def __init__(self, symbols: list, start: pd.Timestamp, end: pd.Timestamp, lambda_risk: float,  # noqa
                 leverage_limit: int, max_weights: float, min_weights: float, **kwargs) -> None:
        """

        :param symbols: list of universe
        :param start: start time
        :param end: end time
        :param lambda_risk: risk aversion ratio
        :param leverage_limit: leverage limit
        :param max_weights: max weight for each ticker
        :param min_weights: min weight for each ticker
        """
        return_forecast = TaReturnsForecast(symbols, start, end)
        risk_model = ReturnsCovRiskModel(symbols, start, end).get_value()

        # can add execution cost, borrow fee
        # tcost_model = cp.TcostModel(half_spread=half_spread)  # different by market, or even more complex by ticker
        # bcost_model = cp.HcostModel(borrow_costs=borrow_costs / 250)  # borrow costs by ticker can be read from broker

        costs = [lambda_risk * cp.FullSigma(risk_model)]  # fit into cp format

        # constraints
        constraints = [
            cp.DollarNeutral(),
            cp.LeverageLimit(leverage_limit),
            cp.MaxWeights(max_weights),
            cp.MinWeights(min_weights)
        ]
        super().__init__(return_forecast, costs, constraints)
