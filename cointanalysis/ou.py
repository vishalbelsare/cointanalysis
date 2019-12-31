import numpy as np


class OrnsteinUhlenbeck:
    """
    Generator of Ornstein-Uhlenbeck process.

    Paramteters
    -----------
    - k : float, default 1.0
        Speed of mean-reversion.
    - mean : float, default 0.0
        Long-term mean of the process.
    - std : float, default 1.0
        Volatility of fluctuation.
    dx[t] = - k * (x - mu) + sigma dW[t]
    where W denotes the Wiener process.

    Notes
    -----
    With parameters, the evolution process is given by

    .. math:: dx_t = - k * (x_t - \mu) + \sigma dW_t

    where :math:`\mu, \sigma` are `mean` and `std`.

    The long-term standard deviation is
    :math:`\sqrt{sigma^2 / (2 k)}`, not `\sigma`.
    """
    def __init__(self, k=1.0, std=1.0, mean=0.0):
        """
        Initialize self.
        """
        self.k = k
        self.std = std
        self.mu = mu

    def array(self, n, d=1, x0=0.0):
        """
        Return np.array of OU process with length n.

        Parameters
        ----------
        - n : int
            Length of a single series
        - d : int, default 1
            Return d arrays.
        - x0 : float
            Initial value.
        """
        def array1d(n, x0):
            t = np.array(range(n))
            dw = np.random.randn(n)
            e = np.exp(- self.k * t)
            I = np.array([[
                np.exp(- self.k * (t - s)) if t >= s else 0
                for s in range(n)
            ] for t in range(n)])

            x = x0 * e + self.mu * (1 - e) + self.sigma * I.dot(dw)
            return x

        if d == 1:
            return array1d(n, x0)
        else:
            return np.array([array1d(n, x0) for _ in range(d)])
