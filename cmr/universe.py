import pandas as pd
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed

from .cache import MEMORY
from .data_loader import load_data, load_symbols


@MEMORY.cache
def build_universe(symbol_pattern: str, start: pd.Timestamp, end: pd.Timestamp, adv_limit: float = 10e6):
    """

    :param symbol_pattern: symbol pattern in regex
    :param start: start timestamp
    :param end: end timestamp
    :param adv_limit:
    :return: list of symbols
    """
    # Load symbols
    symbols = load_symbols(symbol_pattern)

    # Load data
    df = Parallel(n_jobs=8)(delayed(lambda s: load_data(s, start - relativedelta(months=6), end))(s) for s in symbols)
    df = pd.concat(df).sort_values(['time', 'symbol'])

    # Add additional columns
    df['value_traded'] = df.close * df.volume
    df['adv30'] = df.groupby(['symbol']).value_traded.transform(lambda x: x.rolling(10, 1).mean())

    # Filter based on adv limit
    valid_symbols = df[df.adv30 > adv_limit].symbol.unique()
    return sorted(valid_symbols)
