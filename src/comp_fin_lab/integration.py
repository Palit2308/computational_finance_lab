"""
Sometimes the formulas like black scoles are not available in closed form, and we need to use conditional expectations(integrals) to solve them. 
This module contains the code for numerical integration, which is used in some of the models in the comp_fin_lab package.
"""

# Importing necessary libraries
import numpy as np
from scipy.integrate import quad


# This function calculates the price of a European option using integration. 
# The function f is the payoff function of the option, and the integration is performed over the standard normal distribution.

def eu_int_sn(t, f, St, T, r, sigma, a,b):
    """
    Calculates the price of a European option using integration.
    
    Parameters:
    t (float): Current time
    f (function): Payoff function of the option
    St (float): Current stock price
    T (float): Maturity time
    r (float): Risk-free rate
    sigma (float): Volatility
    a (float): Lower limit of integration
    b (float): Upper limit of integration
    
    Returns:
    float: Option price
    """
    def integrand(x):
        return f(St*np.exp((r-sigma**2/2)*(T-t)+sigma*np.sqrt(T-t)*x))*np.exp(-x**2/2)
    return np.exp(-r * (T - t)) / np.sqrt(2 * np.pi) * quad(integrand, a, b)[0]


