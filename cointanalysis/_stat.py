from statsmodels.tsa.stattools import adfuller


class StationarityTester:
    """
    Carry out stationarity test of time-series.

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

    def __init__(self, method="ADF", regression="c"):
        self.method = method
        self.regression = regression

    @property
    def null_hypothesis(self):
        if self.method == "ADF":
            return "unit-root"

    def pvalue(self, x):
        """
        Return p-value of unit-root test.
        Null-hypothesis is that there is a unit root (not stationary).

        Parameters
        ----------
        - X : array, shape (n_samples, )
            Time-series to score p-value.

        Returns
        -------
        pvalue : float
            p-value of the stationarity test.
        """
        if self.method == "ADF":
            if self.regression in ("c", "nc", "ct", "ctt"):
                _, pvalue, _, _, _, _ = adfuller(x, regression=self.regression)
                return pvalue
            else:
                raise ValueError(f"Invalid regression: {self.regression}")
        else:
            raise ValueError(f"Invalid method: {self.method}")

    def is_stationary(self, x, pvalue=0.05):
        """
        Return whether stationarity test implies stationarity.

        Note
        ----
        The name 'is_stationary' may be misleading.
        Strictly speaking, `is_stationary = True` implies that the
        null-hypothesis of the presence of a unit-root has been rejected
        (ADF test) or the null-hypothesis of the absence of a unit-root has
        not been rejected (KPSS test).

        Returns
        -------
        is_stationary : bool
            True may imply the stationarity.
        """
        if self.null_hypothesis == "unit-root":
            return self.pvalue(x) < pvalue
