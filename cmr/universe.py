import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed

from .cache import MEMORY
from .data_loader import load_data, load_symbols


@MEMORY.cache
def build_universe(start: pd.Timestamp, end: pd.Timestamp, adv_limit: float = 10e6):
    """

    :param start: start timestamp
    :param end: end timestamp
    :param adv_limit:
    :return:
    """
    # Load symbols
    symbols = load_symbols("*usd")

    # Load data
    dfs = Parallel(n_jobs=8)(delayed(lambda s: load_data(s, start - relativedelta(months=6), end))(s) for s in symbols)
    df = pd.concat(dfs).sort_values(['time', 'symbol'])

    # Add additional columns
    df['value_traded'] = df.close * df.volume
    df['adv_30d'] = df.groupby(['symbol']).value_traded.transform(lambda x: x.rolling(10, 1).mean())

    # Filter based on adv limit
    valid_symbols = df[df.adv_30d > adv_limit].symbol.unique()
    df = df[df.symbol.isin(valid_symbols)]

    # Drop low quality data
    df = df.drop(columns=['trend_psar_down', 'trend_psar_up']).dropna()

    return df[(df.time >= start) & (df.time <= end)]
