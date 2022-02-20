# crypto-min-resolution

Research challenge for crypto currency pairs at minute resolution

### 1. Hypothesis: Inefficient crypto market hypothesis
#### Rationale (see EDA 1.):
The efficient market hypothesis (EMH) states that financial markets are "informationally efficient" in that the prices
of the traded assets reflect all known information at any given time. But prices vary from day to day despite no new
fundamental information, this involves one aspect that is commonly forgotten among individual traders: liquidity.

Investors that feel overexposed will aggressively hedge or liquidate positions, which will end up affecting the price.
These liquidity demanders are often willing to pay a price to exit their positions, which can result in a profit for
liquidity providers. This ability to profit on information seems to contradict the efficient market hypothesis but forms
the foundation of statistical arbitrage.

### 2. Hypothesis: Correlational crypto assets 
#### Rationale (see EDA 4.):
Statistical arbitrage is an investment strategy that seeks to profit from the narrowing of a gap in the trading prices
of two or more securities. Stat arb involves several different strategies, but all rely on statistically correlational
regularities between various assets in a market that tends toward efficiency.

### 3. Hypothesis: Crypto risk follows the normal distribution 
#### Rationale (see EDA 3. & 5.):
In investing, the standard deviation is used as an indicator of market volatility and thus of risk. The more
unpredictable the price action and the wider the range, the greater the risk. When using standard deviation to measure
risk in the stock market, the underlying assumption is that the majority of price activity follows the pattern of a
normal distribution. In a normal distribution, individual values fall within one standard deviation of the mean, above
or below, 68% of the time. Values are within two standard deviations 95% of the time.

### 4. Hypothesis: Crypto series relative stationarity 
#### Rationale IV (see EDA 6.):
In principle, we do not need to check for stationarity nor correct it when we are using an LSTM. However, if the data is
stationary, it will help with better performance and make it easier for the neural network to learn.
