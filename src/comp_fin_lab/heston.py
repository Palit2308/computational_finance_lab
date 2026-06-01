"""
Heston stochastic volatility path simulation.

The Heston model is defined in continuous time as:

    dS(t) = r S(t) dt + sqrt(gamma(t)) S(t) dW_S(t)

    dgamma(t) = kappa(theta - gamma(t)) dt
                + sigma sqrt(gamma(t)) dW_gamma(t)

with

    corr(dW_S(t), dW_gamma(t)) = rho.

Here gamma(t) is the variance process and sqrt(gamma(t)) is the
stochastic volatility.

This module simulates the Heston model on a discrete time grid using:

1. Euler-type discretization for the variance process.
2. Log-Euler discretization for the stock price process.
3. Cholesky decomposition to generate correlated Brownian shocks.
"""

# Installing necessary imports

import numpy as np

# Subfunctions for simulating heston paths

def random_number_gen(M, I, seed=None):
    """
    Generate independent standard normal random numbers for the Heston model.

    Parameters
    ----------
    M : int
        Number of time steps.
    I : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    rand : ndarray
        Array of independent standard normal shocks with shape (2, M + 1, I).

        The first dimension corresponds to the two Brownian shocks:
        - row 0: stock price shock
        - row 1: variance shock
    """

    rng = np.random.default_rng(seed)
    rand = rng.standard_normal((2, M + 1, I))

    return rand


def _cholesky_from_rho(rho):
    """
    Build the Cholesky matrix for two Brownian motions with correlation rho.

    Parameters
    ----------
    rho : float
        Correlation between stock and variance Brownian shocks.

    Returns
    -------
    cho_matrix : ndarray
        Cholesky decomposition of the 2x2 correlation matrix.
    """

    if rho <= -1 or rho >= 1:
        raise ValueError("rho must be strictly between -1 and 1 for Cholesky decomposition.")

    covariance_matrix = np.zeros((2, 2))
    covariance_matrix[0] = [1.0, rho]
    covariance_matrix[1] = [rho, 1.0]

    cho_matrix = np.linalg.cholesky(covariance_matrix)

    return cho_matrix


def _sde_gamma(gamma0, kappa, theta, sigma, T, M, I, rand, rho):
    """
    Simulate the Heston variance process gamma(t).

    This function follows the same update logic as the original course code,
    but uses gamma notation instead of v notation and builds the Cholesky
    matrix internally from rho.

    Parameters
    ----------
    gamma0 : float
        Initial variance.
    kappa : float
        Speed of mean reversion.
    theta : float
        Long-run variance level.
    sigma : float
        Volatility of variance, also called vol-of-vol.
    T : float
        Time horizon.
    M : int
        Number of time steps.
    I : int
        Number of simulated paths.
    rand : ndarray
        Independent standard normal shocks with shape (2, M + 1, I).
    rho : float
        Correlation between stock and variance Brownian shocks.

    Returns
    -------
    gamma : ndarray
        Simulated variance paths with shape (M + 1, I).
    """

    dt = T / M
    gamma = np.zeros((M + 1, I), dtype=float)
    gamma[0] = gamma0

    sdt = np.sqrt(dt)
    cho_matrix = _cholesky_from_rho(rho)

    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])

        gamma[t] = np.maximum(
            0,
            gamma[t - 1]
            + kappa * (theta - gamma[t - 1]) * dt
            + np.sqrt(gamma[t - 1]) * sigma * ran[1] * sdt,
        )

    return gamma


def heston_paths(S0, r, gamma0, kappa, theta, sigma, rho, T, M, I, seed=None, rand=None):
    """
    Simulate stock price and variance paths under the Heston model.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free interest rate.
    gamma0 : float
        Initial variance.
    kappa : float
        Speed of mean reversion of the variance process.
    theta : float
        Long-run variance level.
    sigma : float
        Volatility of variance, also called vol-of-vol.
    rho : float
        Correlation between stock and variance Brownian shocks.
    T : float
        Time horizon.
    M : int
        Number of time steps.
    I : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility.
    rand : ndarray, optional
        Pre-generated random numbers with shape (2, M + 1, I).
        If None, random numbers are generated internally.

    Returns
    -------
    S : ndarray
        Simulated stock price paths with shape (M + 1, I).
    gamma : ndarray
        Simulated variance paths with shape (M + 1, I).
    """

    dt = T / M
    sdt = np.sqrt(dt)

    if rand is None:
        rand = random_number_gen(M=M, I=I, seed=seed)

    cho_matrix = _cholesky_from_rho(rho)

    gamma = _sde_gamma(gamma0=gamma0, kappa=kappa, theta=theta, sigma=sigma, T=T, M=M, I=I, rand=rand, rho=rho)

    S = np.zeros((M + 1, I), dtype=float)
    S[0] = S0

    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])

        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * gamma[t - 1]) * dt
            + np.sqrt(gamma[t - 1]) * ran[0] * sdt
        )

    return S, gamma


