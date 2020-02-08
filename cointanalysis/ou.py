# import numpy as np


# # TODO exact PhysRevE.54.2084


# class OrnsteinUhlenbeck:
#     """
#     Ornstein-Uhlenbeck process.

#     Paramteters
#     -----------
#     - k : float, default 1.0
#         Speed of mean-reversion.
#     - mean : float, default 0.0
#         Long-term mean of the process.
#     - vola : float, default 1.0
#         Instantaneous volatility.

#     Notes
#     -----
#     The evolution process is given by

#     .. math:: dx_t = - k * (x_t - \\mu) + \\sigma dW_t

#     where :math:`\\mu, \\sigma` are `mean` and `vola`.
#     A wiener process with unit volatility is denoted by :math:`W_t`.

#     The long-term standard deviation is given by
#     :math:`\\sqrt{sigma^2 / (2 k)}`, not `\\sigma`.
#     """
#     def __init__(self, k=1.0, mean=1.0, vola=1.0):
#         """Initialize self."""
#         self.k = k
#         self.mean = mean
#         self.vola = vola

#     def sample(self, size, x0=0.0):
#         """
#         Return np.array of OU process with length n.

#         Parameters
#         ----------
#         - size : int or 2-tuple of ints
#             size of processes.
#         - x0 : float or tuple of floats
#             Initial value.
#         """
#         def _ou1(n, x0):
#             """
#             Return an array with shape (n, ) of a single OU process.
#             """
#             t = np.arange(n)
#             dw = np.random.randn(n)
#             exp = np.exp(- self.k * t)
#             fluc = np.array([[
#                 np.exp(- self.k * (t - s)) if t >= s else 0.0
#                 for s in range(n)
#             ] for t in range(n)])

#             x = x0 * exp + self.mu * (1 - exp) + self.vola * fluc.dot(dw)
#             return x

#         if size.ndim == 1:
#             n, d = size[0], 1
#         if size.ndim == 2:
#             n, d = size
#         if size.ndim > 2:
#             raise ValueError('size should be 1d or 2d-array')

#         if isinstance(x0, np.ndarray):
#             if x0.ndim != 1:
#                 raise ValueError('x0 should be 1darray')
#             if x0.size[0] != d:
#                 raise ValueError('size and x0 mismatch')

#         if d == 1:
#             return _ou1(n, x0)
#         else:
#             return np.array([_ou1(n, x0) for _ in range(d)])
