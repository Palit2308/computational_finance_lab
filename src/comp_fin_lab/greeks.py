"""
Calculates the greeks of options using different methods.
"""
# Import necessary libraries

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from .binom import eu_crr
from .char_fun import heston_char


def eu_crr_delta(g, S0, K, T, r, sigma, M, c=1):
    """
    Computes the CRR delta approximation for a European option.

    The delta is computed using the Pelsser-Vorst idea:
    construct two neighbouring stock values around S0 from the CRR tree,
    then approximate the derivative by a central difference.

    Returns
    -------
    delta : float
        CRR approximation of option delta.
    """

    delta_t = T / M

    beta = (
        c * np.exp(-r * delta_t)
        + np.exp((r + sigma**2) * delta_t)
    ) / 2

    u = beta + np.sqrt(beta**2 - c)
    d = c / u

    S_plus = S0 * u / d
    S_minus = S0 * d / u

    price = eu_crr(
        g=g,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M,
        c=c,
    )

    V_plus = eu_crr(
        g=g,
        S0=S_plus,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M,
        c=c,
    )

    V_minus = eu_crr(
        g=g,
        S0=S_minus,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M,
        c=c,
    )

    delta = (V_plus - V_minus) / (S_plus - S_minus)

    return delta


# Delta of a European call or put option using the Black-Scholes formula

def eu_bs_delta(t, St, K, T, r, sigma, call):
    """
    Calculates the delta of a European call or put option using the Black-Scholes formula.

    Parameters:
    t : float
        Current time (in years).
    St : float
        Current stock price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying stock (annualized).
    call : int
        1 for call option, 0 for put option.

    Returns:
    delta : float
        The delta of the European option at time t.
    """

    d1 = (np.log(St/K)+r*(T-t)+sigma**2/2*(T-t))/(sigma*np.sqrt(T-t))
    if call==1:
        return norm.cdf(d1)
    elif call==0:
        return -norm.cdf(-d1)
    else:
        print('call must be either 1 or 0')

# Delta of a European call or put option using integration

def delta_int_sn(t, f_prime, St, T, r, sigma, a, b):
    """
    Calculates the delta of a European option using the pricing-by-integration approach.

    Parameters:
    t (float): Current time
    f_prime (function): Derivative of the payoff function with respect to S(T)
    St (float): Current stock price
    T (float): Maturity time
    r (float): Risk-free rate
    sigma (float): Volatility
    a (float): Lower limit of integration
    b (float): Upper limit of integration

    Returns:
    float: Option delta
    """

    def integrand(x):
        growth_factor = np.exp(
            (r - sigma**2 / 2) * (T - t)
            + sigma * np.sqrt(T - t) * x
        )

        ST = St * growth_factor

        return f_prime(ST) * growth_factor * np.exp(-x**2 / 2)

    return (
        np.exp(-r * (T - t))
        / np.sqrt(2 * np.pi)
        * quad(integrand, a, b)[0]
    )

def delta_eu_lap_heston(u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho):
    """
    Derivative of the Heston characteristic function with respect to St.

    This is used inside the Laplace inversion formula to compute the
    European option delta.

    Parameters
    ----------
    u : complex
        Complex Fourier-Laplace argument.

    St : float
        Current stock price.

    t : float
        Current time.

    T : float
        Maturity.

    r : float
        Risk-free rate.

    gamt : float
        Current variance at time t.

    kappa : float
        Variance drift level.

    lamb : float
        Mean-reversion speed.

    sig_tilde : float
        Volatility of volatility.

    rho : float
        Correlation between stock and variance Brownian motions.

    Returns
    -------
    complex
        Derivative of the Heston characteristic function with respect to St.
    """

    delta = (
        1j
        * u
        * heston_char(u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho)
        / St
    )

    return delta


def delta_eu_lap_bs(u, St, t, T, r, sig_tilde):
    """
    Derivative of the Black-Scholes characteristic function with respect to St.

    This function keeps the same extended parameter format as the Heston
    delta function for compatibility with the generic Laplace delta routine.

    For Black-Scholes, only sig_tilde is used. It plays the role of sigma.
    The parameters gamt, kappa, lamb, and rho are unused.

    Parameters
    ----------
    u : complex
        Complex Fourier-Laplace argument.

    St : float
        Current stock price.

    t : float
        Current time.

    T : float
        Maturity.

    r : float
        Risk-free rate.

    sig_tilde : float
        Black-Scholes volatility sigma.

    Returns
    -------
    complex
        Derivative of the Black-Scholes characteristic function with respect
        to St.
    """

    delta = 1j * u * np.exp(
        (1j * u - 1) * np.log(St)
        + 1j * u * r * (T - t)
        - (1j * u + u**2) * (sig_tilde**2) / 2 * (T - t)
    )

    return delta


def delta_eu_lap(delta_char_fun, K_arr, R, St, t, T, r, *input_params):
    """
    Compute European option delta using Laplace inversion.

    Parameters
    ----------
    delta_char_fun : callable
        Derivative of the model characteristic function with respect to St.
        It must have signature

            delta_char_fun(u, St, t, T, r, *input_params)

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
        Additional model-specific parameters passed to delta_char_fun.

    Returns
    -------
    deltas : np.ndarray or float
        Delta values for each strike in K_arr. If K_arr is scalar, returns
        a scalar.
    """

    scalar_input = np.isscalar(K_arr)

    K_arr = np.atleast_1d(np.asarray(K_arr, dtype=float))
    deltas = np.zeros_like(K_arr, dtype=float)

    tau = T - t

    if tau <= 0:
        raise ValueError("Maturity T must be larger than current time t.")

    if not (R > 1 or R < 0):
        raise ValueError("Use R > 1 for calls or R < 0 for puts.")

    def f_tilde(z, K):
        """
        Fourier-Laplace transform of the European call/put payoff.

        For call: valid for Re(z) > 1.
        For put:  valid for Re(z) < 0.
        """
        return K ** (1 - z) / (z * (z - 1))

    for i, K in enumerate(K_arr):

        def integrand(u):
            z = R + 1j * u

            value = (
                f_tilde(z, K)
                * delta_char_fun(u - 1j * R, St, t, T, r, *input_params)
            )

            return np.real(value)

        integral_value = quad(integrand, 0, np.inf, limit=250)[0]

        deltas[i] = np.exp(-r * tau) / np.pi * integral_value

    if scalar_input:
        return deltas[0]

    return deltas