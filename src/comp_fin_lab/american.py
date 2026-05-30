"""
American option pricing routines.

Currently includes valuation of a perpetual American put option
in the Black-Scholes model.
"""

import numpy as np
from scipy.integrate import solve_ivp


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