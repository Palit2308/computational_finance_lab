"""
This module contains the payoffs of different options in a simple vectorised format.
"""

# 1. Importing libraries

import numpy as np

# 2. Payoff of a vanilla call option

def vcall(S,K):
    """
    Payoff of a vanilla call option.
    
    Parameters:
    S : array-like
        Underlying asset price at maturity.
    K : float
        Strike price of the option.
    """
    return np.maximum(S-K,0)

# 3. Payoff of a vanilla put option

def vput(S,K):
    """
    Payoff of a vanilla put option.
    
    Parameters:
    S : array-like
        Underlying asset price at maturity.
    K : float
        Strike price of the option.
    """
    return np.maximum(K-S,0)

# 3. Payoff of a vanilla option

def vopt(S,K,call=True):
    """
    Payoff of a vanilla option.
    
    Parameters:
    S : array-like
        Underlying asset price at maturity.
    K : float
        Strike price of the option.
    call : bool, optional
        If True, returns the payoff of a vanilla call option. If False, returns the payoff of a vanilla put option.
    """
    if call:
        return np.maximum(S-K,0)
    else:
        return np.maximum(K-S,0)
    
# 4. Payoff of a power call option

def pcall(S, K, alpha):
    """
    Payoff of a power call option.
    
    Parameters:
    S : array-like
        Underlying asset price at maturity.
    K : float
        Strike price of the option.
    alpha : float
        Power to which the underlying asset price is raised.
    """
    return np.maximum(S**alpha - K, 0)