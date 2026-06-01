import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.bs import bs_paths


# --------------------------------------------------
# Parameters
# --------------------------------------------------

S0 = 100
r = 0.05
sigma = 0.3
T = 1

M0 = 500
M = int(M0 * T)
I = 10000

seed = 42


# --------------------------------------------------
# Simulate Black-Scholes paths
# --------------------------------------------------

S, S_hat, time_grid = bs_paths(
    S0=S0,
    r=r,
    sigma=sigma,
    T=T,
    M=M,
    I=I,
    seed=seed,
)


# --------------------------------------------------
# Basic checks
# --------------------------------------------------

print("Stock path matrix shape:", S.shape)
print("Discounted stock path matrix shape:", S_hat.shape)
print("Time grid shape:", time_grid.shape)

print("Initial stock price:", S[0, 0])
print("Initial discounted stock price:", S_hat[0, 0])

print("Mean terminal stock price:", S[-1].mean())
print("Theoretical E[S(T)]:", S0 * np.exp(r * T))

print("Mean terminal discounted stock price:", S_hat[-1].mean())
print("Theoretical E[S_hat(T)]:", S0)

assert S.shape == (M + 1, I)
assert S_hat.shape == (M + 1, I)
assert time_grid.shape == (M + 1,)

assert np.allclose(S[0], S0)
assert np.allclose(S_hat[0], S0)

assert np.all(np.isfinite(S))
assert np.all(np.isfinite(S_hat))
assert np.all(np.isfinite(time_grid))

assert np.all(S > 0)
assert np.all(S_hat > 0)

# Under Q, the discounted stock price should be approximately a martingale:
# E[S_hat(T)] = S0.
# setting loose se t to 1.5 to avoid test failure due to sampling error.
assert abs(S_hat[-1].mean() - S0) < 1.5

# The undiscounted stock should satisfy:
# E[S(T)] = S0 exp(rT).
# setting loose se t to 2.0 to avoid test failure due to sampling error.
assert abs(S[-1].mean() - S0 * np.exp(r * T)) < 2.0


# --------------------------------------------------
# Log-return diagnostics
# --------------------------------------------------

terminal_log_returns = np.log(S[-1] / S0)

sample_mean = terminal_log_returns.mean()
sample_var = terminal_log_returns.var()

theoretical_mean = (r - 0.5 * sigma**2) * T
theoretical_var = sigma**2 * T

print("\nTerminal log-return diagnostics")
print("Sample mean:", sample_mean)
print("Theoretical mean:", theoretical_mean)
print("Sample variance:", sample_var)
print("Theoretical variance:", theoretical_var)

assert abs(sample_mean - theoretical_mean) < 0.02
assert abs(sample_var - theoretical_var) < 0.02


# --------------------------------------------------
# Reproducibility check
# --------------------------------------------------

S_same, S_hat_same, time_grid_same = bs_paths(
    S0=S0,
    r=r,
    sigma=sigma,
    T=T,
    M=M,
    I=I,
    seed=seed,
)

assert np.allclose(S, S_same)
assert np.allclose(S_hat, S_hat_same)
assert np.allclose(time_grid, time_grid_same)

print("Reproducibility check passed.")


# --------------------------------------------------
# Plot function
# --------------------------------------------------

def plot_paths(n):
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(range(len(S)), S[:, :n])
    ax1.grid()
    ax1.set_title("Black-Scholes Stock Price Paths")
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Timestep")

    ax2.plot(range(len(S_hat)), S_hat[:, :n])
    ax2.grid()
    ax2.set_title("Discounted Black-Scholes Stock Paths")
    ax2.set_ylabel("Discounted price")
    ax2.set_xlabel("Timestep")

    return fig


# --------------------------------------------------
# Save path plot
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

fig = plot_paths(100)

save_path = os.path.join(plot_dir, "bs_stock_and_discounted_paths.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Path plot saved to: {save_path}")


# --------------------------------------------------
# Save terminal distribution diagnostic
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(terminal_log_returns, bins=60, density=True)

plt.axvline(theoretical_mean, linestyle="--", label="Theoretical mean")
plt.axvline(sample_mean, linestyle=":", label="Sample mean")

plt.xlabel("Terminal log return")
plt.ylabel("Density")
plt.title("Black-Scholes Terminal Log-Return Distribution")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "bs_terminal_log_return_distribution.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Distribution plot saved to: {save_path}")

print("Black-Scholes path simulation test passed.")