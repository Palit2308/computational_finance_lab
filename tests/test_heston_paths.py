import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.heston import heston_paths


# --------------------------------------------------
# Parameters
# --------------------------------------------------

gamma0 = 0.04
kappa = 2
sigma = 0.3
theta = 0.04
rho = -0.9

S0 = 100
r = 0.05
T = 1

M0 = 500
M = int(M0 * T)
I = 10000

seed = 42


# --------------------------------------------------
# Simulate Heston paths
# --------------------------------------------------

S, gamma = heston_paths(
    S0=S0,
    r=r,
    gamma0=gamma0,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    rho=rho,
    T=T,
    M=M,
    I=I,
    seed=seed,
)

# gamma is variance.
# sqrt(gamma) is volatility.
V = np.sqrt(gamma)


# --------------------------------------------------
# Basic checks
# --------------------------------------------------

print("Stock path matrix shape:", S.shape)
print("Variance path matrix shape:", gamma.shape)
print("Volatility path matrix shape:", V.shape)

print("Initial stock price:", S[0, 0])
print("Initial variance:", gamma[0, 0])
print("Initial volatility:", V[0, 0])

print("Minimum stock value:", S.min())
print("Minimum variance value:", gamma.min())
print("Maximum variance value:", gamma.max())

# Shape checks
assert S.shape == (M + 1, I)
assert gamma.shape == (M + 1, I)
assert V.shape == (M + 1, I)

# Initial condition checks
assert np.allclose(S[0], S0)
assert np.allclose(gamma[0], gamma0)
assert np.allclose(V[0], np.sqrt(gamma0))

# Finite-value checks
assert np.all(np.isfinite(S))
assert np.all(np.isfinite(gamma))
assert np.all(np.isfinite(V))

# Stock prices should remain positive under log-Euler update
assert np.all(S > 0)

# Variance should be non-negative because of the truncation scheme
assert np.all(gamma >= 0)

# Volatility should also be non-negative
assert np.all(V >= 0)


# --------------------------------------------------
# Reproducibility check
# --------------------------------------------------

S_same, gamma_same = heston_paths(
    S0=S0,
    r=r,
    gamma0=gamma0,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    rho=rho,
    T=T,
    M=M,
    I=I,
    seed=seed,
)

assert np.allclose(S, S_same)
assert np.allclose(gamma, gamma_same)

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
    ax1.set_title("Heston Price paths")
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Timestep")

    ax2.plot(range(len(V)), V[:, :n])
    ax2.grid()
    ax2.set_title("Heston Volatility paths")
    ax2.set_ylabel("Volatility")
    ax2.set_xlabel("Timestep")

    return fig


# --------------------------------------------------
# Save plot
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

fig = plot_paths(100)

save_path = os.path.join(plot_dir, "heston_price_and_volatility_paths.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {save_path}")
print("Heston path simulation test passed.")