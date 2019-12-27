import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, coint


class CointAnalysis(TransformerMixin):
    """
    Parameters
    ----------
    - method : {'AEG'}, default 'AEG'
        Test method.
    - axis : {0, 1, 'PCA'}
        If 0, regress 0th series to 1st series.
        If 1, regress 1st series to 0th series.
        If 'PCA', use PC1.
    - trend : {'c', 'nc'}, default 'c'
        - 'c' : residual = constant
        - 'nc' : residual = 0

    Attributes
    ----------
    - coef_ : array-like, shape (2, )
        The coefficients of prices to make maximally
        stationary spread `spread = X.dot(coef_)`.
    - const_ : float
        The trend constant of the spread.
        For `trend = 'nc'`, it is always 0.0.
    - std_ : float
        The estimated standard deviation of the spread.

    Notes
    -----
    The values `coef_ = (b0, b1)` and `const_ = c` implies that
    ```spread = b0 * X[:, 0] + b1 * X[:, 1]```
    is stationary around c.

    Examples
    --------
    >>> df
               stock1  stock2
    2000-01-01   1234    2468
    2000-01-02   1235    2470
          ....   ....    ....
    >>> X = df.values
    >>> coint = CointAnalysis().fit(X)
    >>> coint.coef_
    (1.0, -0.5)
    >>> coint.coef_.dot(X)
    ...
    """
    def __check_method(self, method):
        if method not in ('AEG', ):
            raise ValueError(f'Invalid method: {method}.')

    def __check_axis(self, axis):
        if axis not in (0, 1, 'PCA'):
            raise ValueError(f'Invalid axis: {axis}.')

    def __check_trend(self, trend):
        if trend not in ('c', 'nc'):
            raise ValueError(f'Invalid trend: {trend}.')

    def __init__(self, method='AEG', axis=0, trend='c'):
        """Initialize self."""
        self.__check_method(method)
        self.method = method
        self.__check_axis(axis)
        self.axis = axis
        self.__check_trend(trend)
        self.trend = trend

    def score(self, X, y=None, method='AEG', stat_method='ADF', stat_pvalue=.05):
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

        # Stationarity test
        stationarity = StationarityTest(method=stat_method, regression='c')
        if stationarity.is_stationary(X[:, 0], stat_pvalue) \
                or stationarity.is_stationary(X[:, 1], stat_pvalue):
            return np.nan

        # Cointegration test
        if self.axis in (0, 1):
            if self.axis == 0:
                X0, X1 = X[:, 0], X[:, 1]
            if self.axis == 1:
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
                pvalue = stat.score(spread)
                return pvalue
            if self.trend == 'nc':
                # pseudo PCA from origin
                rms0 = self.__class__.__rms(X[:, 0])
                rms1 = self.__class__.__rms(X[:, 1])
                spread = X[:, 0] / rms0 - X[:, 1] / rms1
                stat = StationarityTest(method='ADF', regression='nc')
                pvalue = stat.score(spread)
                return pvalue
            raise ValueError

        raise ValueError

    def fit(self, X, y=None):
        """
        Fit by linear regression (for axis = 0, 1) or PCA
        to evaluate self.coef_ and self.const_.

        Parameters
        ----------
        - X : array, shape (n_samples, 2)
            A pair of price data, where n_samples is the number of samples.
        - y : None
            Ignored.

        Returns
        -------
        self
        """
        if self.axis in (0, 1):
            if self.axis == 0:
                X, y = X[:, 0].reshape(-1, 1), X[:, 1]
            if self.axis == 1:
                X, y = X[:, 1].reshape(-1, 1), X[:, 0]

            if self.trend == 'c':
                fi = True
            if self.trend == 'nc':
                fi = False

            reg = LinearRegression(fit_intercept=fi).fit(X, y)

            if self.axis == 0:
                self.coef_ = np.array([- reg.coef_[0], 1.0])
            if self.axis == 1:
                self.coef_ = np.array([1.0, - reg.coef_[0]])

            self.const_ = getattr(reg, 'intercept_', 0.0)
            self.std_ = (y - reg.predict(X)).std()

        if self.axis == 'PCA':
            if self.trend == 'c':
                pca = PCA(n_components=2).fit(X)
                self.coef_ = pca.components_[1]
                self.const_ = pca.mean_.dot(self.coef_)
                self.std_ = np.sqrt(pca.explained_variance_[1])
            if self.trend == 'nc':
                rms0 = self.__class__.__rms(X[:, 0])
                rms1 = self.__class__.__rms(X[:, 1])
                self.coef_ = np.array([1 / rms0, -1 / rms1])
                self.const_ = 0.0
                self.std_ = X.dot(self.coef_).std()

        return self

    def transform(self, X, y=None,
                  subtract_const=True,
                  normalize_std=True):
        """
        Return spread of X.

        Parameters
        ----------
        - X : array, shape (n_samples, 2)
            A pair of price data, where n_samples is the number of samples.
        - y : None
            Ignored.
        - subtract_const : bool, default True
            If True, subtract constant from the spread.
        - normalize_std : bool, default True
            If True, normalize spread so that std of the spread
            becomes unity.

        Returns
        -------
        X_spread : array, shape (n_samples, )
            Spread of two 1d arrays.
        """
        spread = X.dot(self.coef_)
        if subtract_const:
            spread -= self.const_
        if normalize_std:
            if self.std_ < 10E-10:
                raise RuntimeWarning(f'Did not normalize spread since'
                                     f'std {self.std} < 10E-10.')
            else:
                spread /= self.std_
        return spread

    @staticmethod
    def __rms(array):
        """Root-mean square"""
        return np.sqrt(array.mean() ** 2 + array.std() ** 2)


class StationarityTest:
    """
    Parameters
    ----------
    - method : {'ADF'}, default 'ADF'
        Test method.
    - regression : {'c', 'ct', 'ctt', 'nc'}
        Regression method.

    Examples
    --------
    >>> np.random.seed(42)
    >>> gauss = np.random.randn(100)  # stationary
    >>> stat = StationarityAnalisis()
    >>> stat.score(gauss)  # returns p-value
    1.16e-17
    >>> brown = gauss.cumsum()
    >>> stat.score(brown)  # returns p-value
    0.60
    """
    def __check_method(self, method):
        __available = ('ADF', )
        if method not in __available:
            raise ValueError(f'Invalid method: {method}')

    def __check_regression(self, regression):
        __available = ('c', 'ct', 'ctt', 'nc')
        if regression not in __available:
            raise ValueError(f'Invalid regression: {regression}')

    def __init__(self, method='ADF', regression='c'):
        """Initialize self."""
        self.__check_method(method)
        self.__check_regression(regression)

        self.method = method
        self.regression = regression

    def score(self, X, y=None):
        """
        Return p-value of unit-root test.
        Null-hypothesis is that there is a unit root (not stationary).

        Parameters
        ----------
        - X : array, shape (n_samples, 2)
            A pair of price data, where n_samples is the number of samples.
        - y : None
            Ignored.
        """
        if self.method == 'ADF':
            _, pvalue, _, _, _, _ = adfuller(X, regression=self.regression)
            return pvalue

    def is_stationary(self, X, y=None, pvalue=.05):
        return self.score(X) < pvalue
