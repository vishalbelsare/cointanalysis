import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint

from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    assert_all_finite
)

from .stat import StationarityTest


class CointAnalysis(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    - method : {'AEG'}, default='AEG'
        Test method.
        If 'AEG', Augmented Dickey-Fuller test.
    - axis : {'0', '1', 'PCA'}, default=0
        If 0, regress 0th series to 1st series.
        If 1, regress 1st series to 0th series.
        If 'PCA', use PC1.
    - trend : {'c', 'nc'}, default='c'
        - 'c' : residual = constant
        - 'nc' : residual = 0
    - adjust_mean : bool, default True
        If True, subtract mean from the spread.
    - adjust_std : bool, default True
        If True, normalize spread so that std of the spread
        becomes unity.

    Attributes
    ----------
    - coef_ : array, shape (2, )
        The coefficients of prices to make maximally
        stationary spread `spread = X.dot(coef_)`.
    - mean_ : float
        The estimated trend constant of the spread.
        If `trend = 'nc'`, 0.0.
    - std_ : float
        The estimated standard deviation of the spread.

    Notes
    -----
    Given two series are cointegrated, `coef_ = (b0, b1)` and
    `mean_ = c` implies that
    ```spread = b0 * X[:, 0] + b1 * X[:, 1] - c```
    is stationary around 0.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> x0 = np.random.rand(1000).cumsum()
    >>> x1 = 2 * x0 + np.random.rand(1000)
    >>> X = np.array([x0, x1]).T
    >>> coint = CointAnalysis()
    >>> coint.pvalue(X)
    1.92836794233469e-18

    >>> coint.fit(X)
    >>> coint.coef_
    array([-1.99831281,  1. ])
    >>> coint.mean_
    0.011497817617380335
    >>> coint.std_
    0.9488566139638319

    >>> coint.transform(X)
    array([-1.50289357, ...])
    """
    def __init__(self,
                 method='AEG',
                 axis='0',
                 trend='c',
                 adjust_mean=True,
                 adjust_std=True):
        """Initialize self."""
        self.method = method
        self.axis = axis
        self.trend = trend
        self.adjust_mean = adjust_mean
        self.adjust_std = adjust_std

    def fit(self, X, y=None):
        """
        Fit by linear regression (for axis = '0', '1') or PCA (for 'PCA')
        to evaluate self.coef_, self.mean_, and self.std_.

        Parameters
        ----------
        - X : array, shape (n_samples, 2)
            A pair of time-series.
        - y : None
            Ignored.

        Returns
        -------
        self
        """
        X = check_array(X)
        assert_all_finite(X)
        if X.shape[1] != 2:
            raise ValueError('X.shape[1] should be 2.')

        if self.axis in ('0', '1'):
            if self.axis == '0':
                X, y = X[:, 0].reshape(-1, 1), X[:, 1]
            if self.axis == '1':
                X, y = X[:, 1].reshape(-1, 1), X[:, 0]

            if self.trend == 'c':
                fi = True  # fit intercept
            if self.trend == 'nc':
                fi = False

            reg = LinearRegression(fit_intercept=fi).fit(X, y)

            if self.axis == '0':
                self.coef_ = np.array([- reg.coef_[0], 1.0])
            if self.axis == '1':
                self.coef_ = np.array([1.0, - reg.coef_[0]])

            self.mean_ = getattr(reg, 'intercept_', 0.0)
            self.std_ = (y - reg.predict(X)).std()

        if self.axis == 'PCA':
            if self.trend == 'c':
                pca = PCA(n_components=2).fit(X)
                self.coef_ = pca.components_[1]
                self.mean_ = pca.mean_.dot(self.coef_)
                # self.mean_ = X.dot(self.coef_).mean()
                self.std_ = np.sqrt(pca.explained_variance_[1])
            if self.trend == 'nc':
                rms0 = self.__class__.__rms(X[:, 0])
                rms1 = self.__class__.__rms(X[:, 1])
                self.coef_ = np.array([1 / rms0, -1 / rms1])
                self.mean_ = 0.0
                self.std_ = X.dot(self.coef_).std()

        return self

    def __check_method(self, method):
        if method not in ('AEG', ):
            raise ValueError(f'Invalid method: {method}.')

    def __check_axis(self, axis):
        if axis not in ('0', '1', 'PCA'):
            raise ValueError(f'Invalid axis: {axis}.')

    def __check_trend(self, trend):
        if trend not in ('c', 'nc'):
            raise ValueError(f'Invalid trend: {trend}.')

    @staticmethod
    def __rms(array):
        """Root-mean square"""
        return np.sqrt(array.mean() ** 2 + array.std() ** 2)

    def transform(self, X):
        """
        Return spread of X.

        Parameters
        ----------
        - X : array, shape (n_samples, 2)
            A pair of price data, where n_samples is the number of samples.
        - y : None
            Ignored.

        Returns
        -------
        X_spread : array, shape (n_samples, )
            Spread.
        """
        check_is_fitted(self, ['coef_', 'mean_', 'std_'])
        X = check_array(X)
        assert_all_finite(X)
        if X.shape[1] != 2:
            raise ValueError('X.shape[1] should be 2.')

        spread = X.dot(self.coef_)

        if self.adjust_mean:
            spread -= self.mean_
        if self.adjust_std:
            if not self.std_ < 10e-10:
                spread /= self.std_
            else:
                raise RuntimeWarning(f'Did not normalize the spread because'
                                     f'std {self.std_} < 10e-10.')
        return spread

    def pvalue(self, X, y=None, stat_method='ADF', stat_pvalue=.05):
        """
        Return p-value of cointegration test.
        Null-hypothesis is no cointegration.

        Notes
        -----
        If X[:, 0] or X[:, 1] is stationary as such, return np.nan.

        Parameters
        ----------
        - X : array, shape (n_samples, 2)
            A pair of price data.
        - y : None
            Ignored.
        - method : {'AEG'}, default 'AEG'
            Method of cointegration test.
            For self.axis = 'PCA', ignored (use ADF of PCA1).
        - stat_method : {'ADF'}, default 'ADF'
            Method of stationarity test.
        - stat_pvalue : float, default .05
            Threshold of p-value of stationarity test.

        Returns
        -------
        pvalue : float
            p-value of cointegration test.
        """
        assert_all_finite(X)

        # Stationarity test
        stationarity = StationarityTest(method=stat_method, regression='c')
        if stationarity.is_stationary(X[:, 0], stat_pvalue) \
                or stationarity.is_stationary(X[:, 1], stat_pvalue):
            return np.nan

        # Cointegration test
        if self.axis in ('0', '1'):
            if self.axis == '0':
                X0, X1 = X[:, 0], X[:, 1]
            if self.axis == '1':
                X0, X1 = X[:, 1], X[:, 0]
            if self.method == 'AEG':
                _, pvalue, _ = coint(X0, X1, trend=self.trend)
                return pvalue
            raise ValueError

        if self.axis == 'PCA':
            if self.trend == 'c':
                # Usual PCA from the center of distribution
                spread = PCA(n_components=2).fit_transform(X)[:, 1]  # PC1
                stat = StationarityTest(method='ADF', regression='nc')
                pvalue = stat.pvalue(spread)
                return pvalue
            if self.trend == 'nc':
                # pseudo PCA from origin
                rms0 = self.__class__.__rms(X[:, 0])
                rms1 = self.__class__.__rms(X[:, 1])
                spread = X[:, 0] / rms0 - X[:, 1] / rms1
                # TODO method
                stat = StationarityTest(method='ADF', regression='nc')
                pvalue = stat.pvalue(spread)
                return pvalue
            raise ValueError

        raise ValueError
