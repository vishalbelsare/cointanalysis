# CointAnalysis

[![version](https://img.shields.io/pypi/v/cointanalysis.svg)](https://pypi.org/project/cointanalysis/)
[![Build Status](https://travis-ci.com/simaki/cointanalysis.svg?branch=master)](https://travis-ci.com/simaki/cointanalysis)
[![LICENSE](https://img.shields.io/github/license/simaki/cointanalysis)](LICENSE)

Python library for cointegration analysis.

## Functions and Features

- Carry out cointegration test
- Evaluate spread between cointegrated time-series
- Generate cointegrated time-series artificially
- Based on scikit-learn API

## Installation

```sh
$ pip install cointanalysis
```

## What is cointegration?

See [Hamilton's book][hamilton].

## How to use

Let us see how the main class `CointAnalysis` works using two ETFs, [HYG][hyg] and [BKLN][bkln], as examples.
Since they are both connected with liabilities of low-rated companies, these prices behave quite similarly.

![hyg-bkln](./sample/hyg-bkln.png)

### Cointegration test

The method `score` carries out a cointegration test.
The following code gives p-value for null-hypothesis that there is no cointegration.

```python
from cointanalysis import CointAnalysis

hyg = ...   # Fetch historical price of high-yield bond ETF
bkln = ...  # Fetch historical price of bank loan ETF
X = np.array([hyg, bkln]).T

coint = CointAnalysis()
coint.score(X)
# 0.0055
```

The test have rejected the null-hypothesis by p-value of 0.55%, which implies cointegration.

[hyg]: https://www.bloomberg.com/quote/HYG:US
[bkln]: https://www.bloomberg.com/quote/BKLN:US

### Get spread

The method `fit` finds the cointegration equation.

```python
coint = CointAnalysis().fit(X)

coint.coef_
# np.array([-0.18  1.])
coint.const_
# 7.00
coint.std_
# 0.15
```

This means that spread "-0.18 HYG + BKLN" has the mean 7.00 and standard deviation 0.15.

In fact, the prices adjusted with these parameters clarifies the similarities of these ETFs:

![hyg-bkln-adjust](./sample/hyg-bkln-adjust.png)

The time-series of spread is obtained by applying the method `transform` subsequently.
The mean and the standard deviation are automatically adjusted (unless you pass parameters asking not to).

```python
spread = coint.transform(X)
# returns (-0.18 * hyg + 1. * bkln - 7.00) / 0.15

spread = coint.transform(X, adjust_mean=False, adjust_std=False)
# returns -0.18 * hyg + 1. * bkln
```

The method `fit_transform` carries out `fit` and `transform` at once.

```python
spread = coint.fit_transform(X)
```

The result looks like this:

![hyg-bkln-spread](./sample/hyg-bkln-spread.png)

## Acknowledgements

- [statsmodels](https://www.statsmodels.org/)

## References

- [J. D. Hamilton, "Time Series Analysis", (1994)][hamilton].

[hamilton]: https://press.princeton.edu/books/hardcover/9780691042893/time-series-analysis
[statsmodels-aeg]: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html
