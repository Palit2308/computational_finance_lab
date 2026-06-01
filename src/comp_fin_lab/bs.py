"""
Black-Scholes option pricing model implementation.
"""
# Import necessary libraries

import numpy as np
from scipy.stats import norm

# Define the Black-Scholes formula for European call and put options
def eu_bs(t, St, K, T, r, sigma, call):
    """
    Calculates the price of a European call or put option using the Black-Scholes formula.

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
    Vt : float
        The price of the European option at time t.
    """
    d1 = (np.log(St/K)+r*(T-t)+sigma**2/2*(T-t))/(sigma*np.sqrt(T-t))
    d2 = (np.log(St/K)+r*(T-t)-sigma**2/2*(T-t))/(sigma*np.sqrt(T-t))
    if call==1:
        Vt=St*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2)
    elif call==0:
        Vt=K*np.exp(-r*(T-t))*norm.cdf(-d2)-St*norm.cdf(-d1)
    else:
        print('call must be either 1 or 0')
    return Vt
