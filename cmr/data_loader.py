import pathlib

import pandas as pd
import ta
import ta.utils
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed

from .cache import MEMORY

INPUT_PATH = pathlib.Path(__file__).parent.absolute().joinpath("../input")  # TODO： should be in config file formally


def load_symbols(pattern: str = "*usd"):
    """

    :param pattern:
    :return:
    """
    symbols = [p.stem.split('.')[0] for p in INPUT_PATH.glob(f"{pattern}.csv")]
    return symbols


@MEMORY.cache
def load_data(symbol: str, start: pd.Timestamp, end: pd.Timestamp):
    """

    :param symbol: crypto symbol
    :param start: start timestamp
    :param end: end timestamp
    :return:
    """
    path_name = INPUT_PATH.joinpath(symbol + ".csv")

    # Load data
    df = pd.read_csv(path_name, index_col='time', usecols=['time', 'open', 'close', 'high', 'low', 'volume'])

    # Convert timestamp to datetime
    df.index = pd.to_datetime(df.index, unit='ms')

    # Filter to the datetime range
    df = df[(df.index >= start) & (df.index < end)]

    # As mentioned in the description, bins without any change are not recorded.
    # We have to fill these gaps by filling them with the last value until a change occurs.
    df = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).ffill()  # volume will be filled as 0 in agg(), ffill only applies for other fields
    df['ret'] = df.close.pct_change()

    # Add all ta features filling nans values
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume")
    df = df.dropna(axis=1, how='all')

    # Add symbol
    df['symbol'] = symbol

    return df.set_index('symbol', append=True).reset_index()


@MEMORY.cache
def load_market_data(symbols: [str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """

    :param symbols: list of symbols
    :param start: start timestamp
    :param end: end timestamp
    :return:
    """
    # Load data
    df = Parallel(n_jobs=8)(delayed(lambda s: load_data(s, start - relativedelta(months=6), end))(s) for s in symbols)
    df = pd.concat(df).set_index(['time', 'symbol'])
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df[(df.index.get_level_values(0) >= start) & (df.index.get_level_values(0) <= end)]


@MEMORY.cache
def load_features(symbols: [str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """

    :param symbols: list of symbols
    :param start: start timestamp
    :param end: end timestamp
    :return: features dataframe
    """
    # Load data
    df = Parallel(n_jobs=8)(delayed(lambda s: load_data(s, start - relativedelta(months=6), end))(s) for s in symbols)
    df = pd.concat(df).set_index(['time', 'symbol'])

    # Drop non-features columns
    df = df.drop(columns=['open', 'high', 'low', 'close', 'volume'])

    # Drop low quality data
    df = df.drop(columns=['trend_psar_down', 'trend_psar_up']).dropna()

    return df[(df.index.get_level_values(0) >= start) & (df.index.get_level_values(0) <= end)]


@MEMORY.cache
def load_ret(symbols: [str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """

    :param symbols: list of symbols
    :param start: start timestamp
    :param end: end timestamp
    :return: returns dataframe
    """
    df = Parallel(n_jobs=8)(delayed(lambda s: load_data(s, start - relativedelta(months=6), end))(s) for s in symbols)
    df = pd.concat(df).pivot(index='time', columns='symbol', values='ret').fillna(0)
    df['cash'] = 0
    return df[(df.index >= start) & (df.index <= end)]


@MEMORY.cache
def load_cov(symbols: [str], start: pd.Timestamp, end: pd.Timestamp, window: int = 180) -> pd.DataFrame:
    """

    :param symbols: list of symbols
    :param start: start timestamp
    :param end: end timestamp
    :param window: covariance window
    :return:
    """
    df = Parallel(n_jobs=8)(delayed(lambda s: load_data(s, start - relativedelta(months=6), end))(s) for s in symbols)
    df = pd.concat(df).pivot(index='time', columns='symbol', values='ret').fillna(0)
    df['cash'] = 0

    # Risk model in practice
    df = df.rolling(window=window, min_periods=window).cov().dropna()
    return df[(df.index.get_level_values(0) >= start) & (df.index.get_level_values(0) <= end)]
