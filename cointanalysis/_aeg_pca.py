import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
from statsmodels.tsa.stattools import adfuller

from ._utils import rms


def aeg_pca(X0, X1, trend):
    __sqrteps = np.sqrt(np.finfo(np.double).eps)

    # Compute residual
    if trend == "nc":
        # pseudo PCA from origin
        rms0 = rms(X0)
        rms1 = rms(X1)
        residual = X0 / rms0 - X1 / rms1
        collinearity = 1.0 - residual.std() / (X0 / rms1 + 1 / rms0).std()
    if trend == "c":
        X = np.array([X0, X1]).T
        pca = PCA(n_components=2).fit(X)
        residual = pca.transform(X)[:, 1]  # PC1
        collinearity = pca.explained_variance_ratio_[0]

    # Get ADF statistics
    if collinearity < 1.0 - __sqrteps:
        adf_stat = adfuller(residual, regression="nc")[0]
    else:
        adf_stat = -np.inf

    # Get pvalue
    pvalue = mackinnonp(adf_stat, regression=trend, N=1)

    # Get critical values
    if trend == "nc":
        crit = [np.nan, np.nan, np.nan]
    if trend == "c":
        crit = mackinnoncrit(N=1, regression=trend, nobs=X0.shape[0] - 1)

    return adf_stat, pvalue, crit
