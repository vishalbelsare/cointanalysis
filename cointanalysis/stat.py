from statsmodels.tsa.stattools import adfuller


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
    >>> stat.pvalue(gauss)  # returns p-value
    1.16e-17
    >>> brown = gauss.cumsum()
    >>> stat.pvalue(brown)  # returns p-value
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

    def pvalue(self, X, y=None):
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
        return self.pvalue(X) < pvalue
