import pathlib

import pandas as pd
from joblib import Parallel, delayed

from .cache import MEMORY

__all__ = ['load_symbols', 'load_data']

INPUT_PATH = pathlib.Path(__file__).parent.absolute().joinpath("../input")  # TODO： should be in config file formally


def load_symbols(pattern: str = "*usd"):
    """

    :param pattern:
    :return:
    """
    symbols = [p.stem.split('.')[0] for p in INPUT_PATH.glob(f"{pattern}.csv")]
    return symbols


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
    })

    # Add symbol
    df['symbol'] = symbol

    return df.set_index('symbol', append=True).reset_index()


