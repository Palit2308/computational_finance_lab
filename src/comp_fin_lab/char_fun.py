# Code to generate the characteristic function of a random variable according to the model being used. This is used in laplace and fourier transforms based option pricing.

import numpy as np


# bs characteristic function

def bs_char(u, St, t, T, r, sigma):
    """
    Characteristic function of the Black-Scholes model.
    Parameters
    ----------
    u : complex
        The argument of the characteristic function.
    St : float
        The spot price at time t.
    t : float
        The current time.
    T : float
        The maturity time.
    r : float
        The risk-free rate.
    sigma : float
        The volatility of the underlying asset.
    Returns
    -------
    complex
        The value of the characteristic function at u.
    """
    tau = T - t

    mean = np.log(St) + (r - 0.5 * sigma**2) * tau
    variance = sigma**2 * tau

    return np.exp(1j * u * mean - 0.5 * variance * u**2)

# heston characteristic function according to Albrecher, Mayer, Schoutens and Tistaert (2007). Stable implimentation.

def heston_char(u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho):
    """
    Characteristic function of the Heston model. See Albrecher, Mayer, Schoutens and Tistaert (2007) for details. 
    This is a stable implementation that avoids numerical issues for large maturities and volatilities.

    Parameters
    ----------
    u : complex
        The argument of the characteristic function.
    St : float
        The spot price at time t.
    t : float
        The current time.
    T : float
        The maturity time.
    r : float
        The risk-free rate.
    gamt : float
        Current variance at time t.
    kappa : float
        Variance drift level. In the parameterization
        dgamma_t = (kappa - lamb * gamma_t)dt
                + sig_tilde * sqrt(gamma_t)dW_t,
        the long-run variance is kappa / lamb.
    lamb : float
        Mean-reversion speed of the variance process.
    sig_tilde : float
        The volatility of volatility.
    rho : float
        The correlation between the Brownian motions.

    Returns
    -------
    complex
        The value of the characteristic function at u.
    """
    B = rho * sig_tilde * u * 1j - lamb
    D = np.sqrt(B**2 + sig_tilde**2 * u * (u + 1j))
    lnG = np.log(B * (np.exp(-D * (T - t)) - 1) / (2 * D) + (np.exp(-D * (T - t)) + 1) / 2)
    psi_0 = u * 1j * r * (T - t) - kappa / sig_tilde**2 * ((B + D) * (T - t) + 2 * lnG) 
    psi_1 = u * 1j
    psi_2 = u * (u + 1j) * (np.exp(-D * (T - t)) - 1) / (B * (np.exp(-D * (T - t)) - 1) + D * (np.exp(-D * (T - t)) + 1))
    return np.exp(psi_0 + psi_1 * np.log(St) + psi_2 * gamt)