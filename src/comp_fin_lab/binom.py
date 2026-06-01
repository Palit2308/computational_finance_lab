"""
Binomial tree implementation for option pricing, using the Cox-Ross-Rubinstein (CRR) model.
"""
# 1. Importing libraries

import numpy as np

# 2. CRR binomial model for european option pricing, anchored at u*d=c

def eu_crr(g, S0, K, T, r, sigma, M, c=1):
    
    """
    This function takes an initial stock price, moves forward according to CRR model, and then discounts the payoff at maturity back to the present value. 
    The tree is anchored at u*d=c, where c is a parameter that can be adjusted to improve the accuracy of the tree for certain types of options (e.g., deep in-the-money or out-of-the-money options). 
    The function returns the price of the option at time 0.

    Parameters:
    g : function
        Payoff function of the option, which takes an array of underlying asset prices at maturity and returns an array of payoffs.
    S0 : float
        Initial stock price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    M : int
        Number of time steps.
    c : float, optional
        Parameter for anchoring the tree (default is 1).
    
    Returns:
    V0 : float
        The price of the European option at time 0.
    """

    delta_t = T/M
    beta = (c*np.exp(-r*delta_t) + np.exp((r+sigma**2)*delta_t))/2
    u = beta+np.sqrt(beta**2 - c)
    d = c/u
    q = (np.exp(r*delta_t)-d)/(u-d)
    S = np.zeros((M+1, M+1))
    V = np.zeros_like(S)
    
    for i in range(0, M+1):
        for j in range(0,i+1):
            S[j,i] = S0*u**j*d**(i-j)

    V[:,M] = g(S[:,M],K) # lambda function which runs through all the rows in a given column of S

    for i in range(M-1,-1, -1):
        for j in range(i,-1,-1):
            V[j,i]= np.exp(-r*delta_t)*(q*V[j+1,i+1]+(1-q)*V[j,i+1])
    
    return V[0,0]


# Same function useful for visualisation of the tree. 

def eu_crr_lattice(g, S0, K, T, r, sigma, M, c=1):
    """
    Build the full CRR stock price and option value lattices.

    This is useful for diagnostics and visualisation.

    Returns
    -------
    S : ndarray
        Stock price tree.
    V : ndarray
        Option value tree.
    delta_t : float
        Time step size.
    """

    delta_t = T / M

    beta = (c * np.exp(-r * delta_t) + np.exp((r + sigma**2) * delta_t)) / 2
    u = beta + np.sqrt(beta**2 - c)
    d = c / u
    q = (np.exp(r * delta_t) - d) / (u - d)

    S = np.zeros((M + 1, M + 1))
    V = np.zeros_like(S)

    for i in range(0, M + 1):
        for j in range(0, i + 1):
            S[j, i] = S0 * u**j * d**(i - j)

    V[:, M] = g(S[:, M], K)

    for i in range(M - 1, -1, -1):
        for j in range(i, -1, -1):
            V[j, i] = np.exp(-r * delta_t) * (
                q * V[j + 1, i + 1] + (1 - q) * V[j, i + 1]
            )

    return S, V, delta_t