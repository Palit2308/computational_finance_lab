"""
Calculates the greeks of options using different methods.
"""
# Import necessary libraries

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from .binom import eu_crr


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