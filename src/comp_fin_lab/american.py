"""
American option pricing routines.

Currently includes valuation of a perpetual American put option
in the Black-Scholes model.
"""

# Import necessary libraries

import numpy as np
from scipy.integrate import solve_ivp


# American option pricing functions

def am_crr(S_ini, K, T, r, sigma, N, opttype, c=1):

    """
    Price an American option using a binomial tree.

    Parameters
    ----------
    S_ini : float
        Initial stock price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    N : int
        Number of time steps.
    opttype : str
        Option type: "C" for call, "P" for put.

    c : float
        Parameter for anchoring the tree (default is 1).
    
    Returns
    -------
    price : float
        Option price at time 0.
    C : ndarray
        Matrix of option prices at each node.
    S : ndarray
        Matrix of underlying prices at each node.
    """

    opttype = opttype.upper()

    if opttype not in ["C", "P"]:
        raise ValueError("opttype must be either 'C' for call or 'P' for put.")

    dt = T / N  # Define time step
    beta = (c*np.exp(-r*dt) + np.exp((r+sigma**2)*dt))/2
    u = beta+np.sqrt(beta**2 - c)
    d = c/u
    p = (np.exp(r * dt) - d) / (u - d)  # risk neutral probs
    C = np.zeros([N + 1, N + 1])  # call prices
    S = np.zeros([N + 1, N + 1])  # underlying price

    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
        if opttype == "C":
            C[N, i] = np.maximum(S[N, i] - K, 0)
        else:
            C[N, i] = np.maximum(K - S[N, i], 0)

    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            C[j, i] = np.exp(-r * dt) * (
                p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i]
            )  # Computing the European option prices
            S[j, i] = (
                S_ini * (u ** (i)) * (d ** (j - i))
            )  # Underlying evolution for each node
            if opttype == "C":
                C[j, i] = np.maximum(
                    C[j, i], S[j, i] - K
                )  # Decision between the European option price and the payoff from early-exercise
            else:
                C[j, i] = np.maximum(
                    C[j, i], K - S[j, i]
                )  # Decision between the European option price and the payoff from early-exercise

    return C[0, 0], C, S


#  Perpetual American put option pricing function

def perp_am_put(K, r, sigma, S_min=1e-6, S_max=200, n_grid=200, S_grid=None):
    """
    Price a perpetual American put option in the Black-Scholes model.

    The perpetual American put has no maturity date. Its value is computed
    by splitting the stock-price domain into:

    1. Exercise region: S <= x_star
       Value equals immediate exercise payoff: K - S.

    2. Continuation region: S > x_star
       Value solves the stationary Black-Scholes ODE.

    Parameters
    ----------
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    S_min : float, optional
        Lower bound of stock-price grid.
    S_max : float, optional
        Upper bound of stock-price grid.
    n_grid : int, optional
        Number of grid points.
    S_grid : array-like, optional
        Custom stock-price grid. If given, S_min, S_max and n_grid are ignored.

    Returns
    -------
    S_grid : ndarray
        Stock-price grid.
    v_grid : ndarray
        Perpetual American put values on the grid.
    x_star : float
        Optimal exercise boundary.
    """

    if r <= 0:
        raise ValueError("r must be positive for this perpetual American put formula.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if S_grid is None:
        S_grid = np.linspace(S_min, S_max, n_grid)
    else:
        S_grid = np.asarray(S_grid, dtype=float)

    v_grid = np.zeros_like(S_grid)

    # Optimal exercise boundary for the perpetual American put
    x_star = (2 * K * r) / (2 * r + sigma**2)

    # Exercise region: value equals payoff
    exercise_mask = S_grid <= x_star
    v_grid[exercise_mask] = np.maximum(K - S_grid[exercise_mask], 0.0)

    # Continuation region: solve the ODE
    continuation_mask = S_grid >= x_star
    s_cont = S_grid[continuation_mask]

    if len(s_cont) > 0:
        def ode_fun(x, v):
            dv1 = v[1]
            dv2 = ((2 * r) / (sigma**2 * x**2)) * (v[0] - x * v[1])
            return [dv1, dv2]

        # Boundary conditions at x_star:
        # v(x_star) = K - x_star
        # v'(x_star) = -1
        v1 = np.maximum(K - x_star, 0.0)
        v2 = -1.0

        sol = solve_ivp(
            fun=ode_fun,
            t_span=(x_star, S_grid[-1]),
            y0=[v1, v2],
            t_eval=s_cont,
        )

        v_grid[continuation_mask] = sol.y[0]

    return S_grid, v_grid, x_star