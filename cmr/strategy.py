import cvxportfolio as cp

from .returns import TaReturnsForecast
from .risk import ReturnsCovRiskModel


class CryptoStatArb(cp.SinglePeriodOpt):

    def __init__(self, return_forecast, costs,  # noqa
                 leverage_limit: int, max_weights: float, min_weights: float, **kwargs) -> None:
        """

        :param symbols: list of universe
        :param start: start time
        :param end: end time
        :param leverage_limit: leverage limit
        :param max_weights: max weight for each ticker
        :param min_weights: min weight for each ticker
        """
        # constraints
        constraints = [
            cp.DollarNeutral(),
            cp.LeverageLimit(leverage_limit),
            cp.MaxWeights(max_weights),
            cp.MinWeights(min_weights)
        ]
        super().__init__(return_forecast, costs, constraints)
