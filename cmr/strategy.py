import cvxportfolio as cp


class CryptoStatArb(cp.SinglePeriodOpt):

    def __init__(self, return_forecast, costs,  # noqa
                 leverage_limit: int, max_weights: float, min_weights: float, **kwargs) -> None:
        """

        :param return_forecast: returns forecast
        :param costs: risk model, transaction model or carry cost model
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


class CryptoLongOnly(cp.SinglePeriodOpt):

    def __init__(self, return_forecast, costs,  # noqa
                 leverage_limit: int, max_weights: float, **kwargs):
        """

        :param return_forecast: returns forecast
        :param costs: risk model, transaction model or carry cost model
        :param leverage_limit: leverage limit
        :param max_weights: max weight for each ticker
        :param kwargs:
        """
        # constraints
        constraints = [
            cp.LongOnly(),
            cp.LeverageLimit(leverage_limit),
            cp.MaxWeights(max_weights)
        ]
        super().__init__(return_forecast, costs, constraints)
