"""
Barrier options are a type of exotic option where the payoff depends on whether the underlying asset's price reaches a certain level (the barrier) during the option's life. 
"""

# 1. Importing libraries

import numpy as np

# 2. Down-and-out call option pricing using the CRR binomial tree model

def down_and_out_crr(g, S0, K, T, B, r, sigma, M, c=1):
    
    """
    This function takes an initial stock price, moves forward according to CRR model, and then discounts the payoff at maturity back to the present value, considering the barrier condition. 
    The tree is anchored at u*d=c, where c is a parameter that can be adjusted to improve the accuracy of the tree for certain types of options (e.g., deep in-the-money or out-of-the-money options). 
    The function returns the price of the option at time 0.

    Parameters:
    g : function
        Payoff function of the option, which takes an array of underlying asset prices at maturity and a strike price, and returns an array of payoffs.
    S0 : float
        Initial stock price.
    K : float
        Strike price of the option.
    T : float
        Time to maturity.
    B : float
        Barrier level. If the underlying asset price falls below this level at any time during the option's life, the option becomes worthless.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    M : int
        Number of time steps.
    c : float, optional
        Parameter for anchoring the tree (default is 1).
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
            if S[j,i] >= B:
                V[j,i]= np.exp(-r*delta_t)*(q*V[j+1,i+1]+(1-q)*V[j,i+1])
            else:
                V[j,i]=0
    
    return V[0,0]