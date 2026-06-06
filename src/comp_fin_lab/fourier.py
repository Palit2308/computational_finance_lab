# Fourier transform based pricing of European options

import numpy as np


def eu_fourier(char_fun, K_arr, R, St, t, T, r, M, N, *input_params):
    """
    Price European options using Fourier inversion with FFT.

    Parameters
    ----------
    char_fun : callable
        Characteristic function of log(S_T), with signature

            char_fun(u, St, t, T, r, *input_params)

    K_arr : array-like
        Strike prices where the option should be priced.

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

    M : float
        Fourier domain truncation parameter.

    N : int
        Number of FFT grid points.

    *input_params :
        Additional parameters passed to char_fun.
        For Heston, for example:

            gamt, kappa, lamb, sig_tilde, rho

    Returns
    -------
    prices : np.ndarray
        Option prices for each strike in K_arr.
    """

    K_arr = np.asarray(K_arr, dtype=float)

    if np.any(K_arr <= 0):
        raise ValueError("All strikes must be strictly positive.")

    tau = T - t

    if tau <= 0:
        raise ValueError("Maturity T must be larger than current time t.")

    if not (R > 1 or R < 0):
        raise ValueError("Use R > 1 for calls or R < 0 for puts.")

    if N <= 0:
        raise ValueError("N must be strictly positive.")

    if M <= 0:
        raise ValueError("M must be strictly positive.")

    N = int(N)

    def f_tilde(z, K):
        """
        Fourier-Laplace transform of the European call/put payoff.

        For call: valid for Re(z) > 1.
        For put:  valid for Re(z) < 0.
        """
        return K ** (1 - z) / (z * (z - 1))

    def g(u):
        return (
            f_tilde(R + 1j * u, 1)
            * char_fun(u - 1j * R, St, t, T, r, *input_params)
        )

    # --------------------------------------------------
    # Sort strikes for interpolation, then restore order
    # --------------------------------------------------

    sort_idx = np.argsort(K_arr)
    K_sorted = K_arr[sort_idx]

    # --------------------------------------------------
    # FFT grid
    # --------------------------------------------------

    n = np.arange(1, N + 1)

    kappa_0 = np.log(K_sorted[0])
    delta = M / N

    kappa_m = kappa_0 + (n - 1) * (2 * np.pi) / M

    u_n = (n - 0.5) * delta

    x = (
        g(u_n)
        * delta
        * np.exp(-1j * (n - 1) * delta * kappa_0)
    )

    x_hat = np.fft.fft(x)

    V_kappa_m = (
        np.exp(-r * tau + (1 - R) * kappa_m)
        / np.pi
        * np.real(x_hat * np.exp(-0.5j * delta * kappa_m))
    )

    K_grid = np.exp(kappa_m)

    if K_sorted[-1] > K_grid[-1]:
        raise ValueError(
            "FFT strike grid does not cover the largest requested strike. "
            "Increase N or decrease M."
        )

    prices_sorted = np.interp(K_sorted, K_grid, V_kappa_m)

    # Restore original strike order
    prices = np.empty_like(prices_sorted)
    prices[sort_idx] = prices_sorted

    return prices