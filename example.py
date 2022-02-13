import json
import logging
import sys

import pandas as pd
import cvxportfolio as cp
from cmr.data_loader import load_ret
from cmr.universe import build_universe
from cmr.strategy import CryptoStatArb

if __name__ == '__main__':
    # setup logger
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    # read config and create start/end time
    cfg = json.load(open(sys.argv[1]))
    print(cfg)
    start = pd.Timestamp(2019, 1, 1)
    end = pd.Timestamp(2022, 1, 30)

    # build universe
    symbols = build_universe(cfg['symbol_pattern'], start, end, adv_limit=cfg['adv_limit'])

    # build strategy
    optimizer = CryptoStatArb(symbols, start, end, **cfg['opt_kwargs'])
    signals = optimizer.return_forecast.returns
    print(signals)

    # back-test
    returns = load_ret(symbols, start, end).fillna(0)
    market_sim = cp.MarketSimulator(returns, [], cash_key='cash')
    initial_portfolio = pd.Series(index=returns.columns, data=0)
    initial_portfolio.loc['cash'] = 10e6
    result = market_sim.run_backtest(initial_portfolio, signals.index[0], signals.index[-1], policy=optimizer)

    # performance analytics
    result.summary()
