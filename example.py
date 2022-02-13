import pandas as pd

from cmr.universe import build_universe

if __name__ == '__main__':
    univ = build_universe(pd.Timestamp(2019, 1, 1), pd.Timestamp(2022, 1, 31))
    print(univ)
