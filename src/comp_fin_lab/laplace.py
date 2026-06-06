# Laplace transform of a EU Option

import numpy as np
from scipy.integrate import quad


def eu_lap(char_fun, K_arr, R, St, t, T, r, *input_params):
    """
    Price of vanilla European options using Laplace inversion.

    Parameters
    ----------
    char_fun : callable
        Characteristic function of log(S_T), with signature

            char_fun(u, St, t, T, r, *input_params)

    K_arr : array-like
        Strike prices.

    R : float
        Real part of the complex integration contour.
        Use R > 1 for calls.
        Use R < 0 for puts.

    St : float
        Current stock price.

    t : float
        Current time.

    T : float
        Maturity.

    r : float
        Risk-free rate.

    *input_params :
        Additional parameters passed to char_fun.
        For Heston, for example:

            gam0, kappa, lamb, sig_tilde, rho

    Returns
    -------
    prices : np.ndarray
        Option prices for each strike in K_arr.
    """

    K_arr = np.asarray(K_arr, dtype=float)
    prices = np.zeros_like(K_arr, dtype=float)

    tau = T - t

    if tau <= 0:
        raise ValueError("Maturity T must be larger than current time t.")

    if R == 0 or R == 1:
        raise ValueError("R cannot be 0 or 1 because f_tilde has poles there.")
    
    def f_tilde(z, K):
        """
        Fourier-Laplace transform of the European call/put payoff.

        For call: valid for Re(z) > 1.
        For put:  valid for Re(z) < 0.

        Same formula works for both, but the integration line R changes.
        """
        return K ** (1 - z) / (z * (z - 1))

    for i, K in enumerate(K_arr):

        def integrand(u):
            z = R + 1j * u

            value = (
                f_tilde(z, K)
                * char_fun(u - 1j * R, St, t, T, r, *input_params)
            )

            return np.real(value)

        integral_value = quad(integrand, 0, np.inf, limit=250)[0]

        prices[i] = np.exp(-r * tau) / np.pi * integral_value

    return prices