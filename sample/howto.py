import numpy as np
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt

from cointanalysis import CointAnalysis


def fetch_etf(ticker):
    return pandas_datareader.data.DataReader(
        ticker, 'yahoo', '2012-01-01', '2018-12-31'
    )['Adj Close']


def plot_prices(hyg, bkln):
    plt.figure(figsize=(16, 4))

    plt.title('HYG and BKLN')
    plt.plot(100 * hyg / hyg[0], label='HYG (2012-01-01 = 100)', linewidth=1)
    plt.plot(100 * bkln / bkln[0], label='BKLN (2012-01-01 = 100)', linewidth=1)

    plt.legend()
    plt.savefig('hyg-bkln.png', bbox_inches="tight", pad_inches=0.1)

def plot_adjust(hyg, bkln, coint):
    plt.figure(figsize=(16, 4))

    hyg_ = (-coint.coef_[0]) * hyg + coint.const_

    plt.title('HYG and BKLN, adjusted')
    plt.plot(hyg_,
             label=f'{-coint.coef_[0]:.2f} * HYG + {coint.const_:.2f}',
             linewidth=1)
    plt.plot(bkln, label='BKLN', linewidth=1)

    plt.legend()
    plt.savefig('hyg-bkln-adjust.png', bbox_inches="tight", pad_inches=0.1)


def plot_spread(spread):
    plt.figure(figsize=(16, 4))

    plt.title('Spread between HYG and BKLN')
    plt.plot(spread, linewidth=1)

    plt.legend()
    plt.savefig('hyg-bkln-spread.png', bbox_inches="tight", pad_inches=0.1)


def main():
    hyg = fetch_etf('HYG')
    bkln = fetch_etf('BKLN')

    plot_prices(hyg, bkln)

    X = np.array([hyg, bkln]).T
    coint = CointAnalysis()

    # score
    pvalue = coint.score(X)
    print(f'pvalue: {pvalue}')

    # fit, transform
    spread = pd.Series(coint.fit_transform(X), index=hyg.index)
    print(f'coef: {coint.coef_}')
    print(f'const: {coint.const_}')
    print(f'std: {coint.std_}')

    plot_adjust(hyg, bkln, coint)
    plot_spread(spread)


if __name__ == '__main__':
    main()
