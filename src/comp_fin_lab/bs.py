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

# Simulating stock price paths under the Black-Scholes model

def bs_paths(S0, r, sigma, T, M, I, seed=None):
    """
    Simulate Black-Scholes stock price paths.

    The simulation first constructs the discounted stock price process

        S_hat(t) = exp(-rt) S(t)

    under the risk-neutral measure, and then undiscounts it internally to
    obtain the stock price process S(t).

    Under the risk-neutral Black-Scholes model:

        dS(t) = r S(t) dt + sigma S(t) dW(t)

    The exact solution is:

        S(t) = S0 exp((r - 0.5 sigma^2)t + sigma W(t))

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Constant volatility.
    T : float
        Time horizon.
    M : int
        Number of time steps.
    I : int
        Number of simulated paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray
        Undiscounted Black-Scholes stock price paths with shape (M + 1, I).
    S_hat : ndarray
        Discounted stock price paths with shape (M + 1, I).
    time_grid : ndarray
        Time grid with shape (M + 1,).
    """

    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if T <= 0:
        raise ValueError("T must be positive.")

    if M <= 0:
        raise ValueError("M must be positive.")

    if I <= 0:
        raise ValueError("I must be positive.")

    dt = T / M
    time_grid = np.linspace(0, T, M + 1)

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal((M, I))
    dW = np.sqrt(dt) * Z

    W = np.zeros((M + 1, I), dtype=float)
    W[1:] = np.cumsum(dW, axis=0)

    # Discounted stock process under the risk-neutral measure
    S_hat = S0 * np.exp(
        -0.5 * sigma**2 * time_grid[:, None]
        + sigma * W
    )

    # Undiscount internally
    bank_account = np.exp(r * time_grid)

    S = bank_account[:, None] * S_hat

    return S, S_hat, time_grid

# BS Pde approximation

def bs_pde(S0, K, T, r, sigma, call=1, S_max=None, S_steps=300,t_steps=20000):
    """
    Price a European call or put option by solving the Black-Scholes PDE
    using an explicit finite-difference scheme.

    The Black-Scholes PDE is solved in time-to-maturity tau = T - t:

        dV/dtau = 0.5 sigma^2 S^2 d2V/dS2
                  + r S dV/dS
                  - r V

    with initial condition at tau = 0:

        V(0, S) = payoff(S)

    Parameters
    ----------
    S0 : float
        Initial stock price where the option price is evaluated.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    call : int
        1 for call option, 0 for put option.
    S_max : float, optional
        Upper boundary of stock-price grid.
        If None, set to max(4*K, 4*S0).
    S_steps : int, optional
        Number of stock-price grid steps.
    t_steps : int, optional
        Number of time steps in time-to-maturity.

    Returns
    -------
    price : float
        Interpolated option price at S0.
    V : ndarray
        PDE value grid with shape (t_steps + 1, S_steps + 1).
        Rows correspond to time-to-maturity tau.
        Columns correspond to stock-price grid points.
    S_grid : ndarray
        Stock-price grid.
    tau_grid : ndarray
        Time-to-maturity grid.
    """

    if call not in [0, 1]:
        raise ValueError("call must be 1 for call option or 0 for put option.")

    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if T <= 0:
        raise ValueError("T must be positive.")

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if S_steps <= 2:
        raise ValueError("S_steps must be greater than 2.")

    if t_steps <= 0:
        raise ValueError("t_steps must be positive.")

    if S_max is None:
        S_max = max(4 * K, 4 * S0)

    if S_max <= S0:
        raise ValueError("S_max should be larger than S0.")

    # --------------------------------------------------
    # Build grids
    # --------------------------------------------------

    S_grid = np.linspace(0, S_max, S_steps + 1)
    tau_grid = np.linspace(0, T, t_steps + 1)

    dS = S_grid[1] - S_grid[0]
    dtau = tau_grid[1] - tau_grid[0]

    # --------------------------------------------------
    # Explicit scheme stability check
    # --------------------------------------------------
    # The explicit scheme can become unstable if dtau is too large.
    # This is a practical safety check.

    stability_bound = 0.9 / (sigma**2 * S_steps**2 + abs(r))

    if dtau > stability_bound:
        raise ValueError(
            "Explicit finite-difference scheme may be unstable. "
            "Increase t_steps or reduce S_steps. "
            f"Current dtau={dtau:.6e}, suggested maximum={stability_bound:.6e}."
        )

    # --------------------------------------------------
    # Initialize value grid
    # --------------------------------------------------

    V = np.zeros((t_steps + 1, S_steps + 1))

    if call == 1:
        V[0, :] = np.maximum(S_grid - K, 0.0)
    else:
        V[0, :] = np.maximum(K - S_grid, 0.0)

    # Interior grid index
    i = np.arange(1, S_steps)

    # Explicit finite-difference coefficients
    alpha = 0.5 * dtau * (sigma**2 * i**2 - r * i)
    beta = 1.0 - dtau * (sigma**2 * i**2 + r)
    gamma = 0.5 * dtau * (sigma**2 * i**2 + r * i)

    # --------------------------------------------------
    # Time stepping forward in tau
    # --------------------------------------------------

    for n in range(0, t_steps):
        V[n + 1, 1:S_steps] = (
            alpha * V[n, 0:S_steps - 1]
            + beta * V[n, 1:S_steps]
            + gamma * V[n, 2:S_steps + 1]
        )

        tau = tau_grid[n + 1]

        # Boundary conditions
        if call == 1:
            V[n + 1, 0] = 0.0
            V[n + 1, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[n + 1, 0] = K * np.exp(-r * tau)
            V[n + 1, -1] = 0.0

    # Interpolate price at S0 from the final tau = T row
    price = np.interp(S0, S_grid, V[-1, :])

    return price, V, S_grid, tau_grid