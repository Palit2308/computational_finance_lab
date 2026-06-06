import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.char_fun import heston_char, bs_char
from comp_fin_lab.laplace import eu_lap


# --------------------------------------------------
# Parameters
# --------------------------------------------------

R = 2
K_arr = np.linspace(50, 300, 100)

r = 0.02
gam0 = 0.3**2
kappa = 0.3**2
lamb = 2.5
sig_tilde = 0.3
rho = -0.5

t = 0
T = 1
S0 = 120


# --------------------------------------------------
# Compute Heston call prices using Laplace inversion
# --------------------------------------------------

call_Heston_Laplace = eu_lap(
    heston_char,
    K_arr,
    R,
    S0,
    t,
    T,
    r,
    gam0,
    kappa,
    lamb,
    sig_tilde,
    rho,
)


# --------------------------------------------------
# Basic sanity checks
# --------------------------------------------------

tau = T - t

# 1. Output should have same shape as strike array
assert call_Heston_Laplace.shape == K_arr.shape

# 2. Prices should be finite
assert np.all(np.isfinite(call_Heston_Laplace))

# 3. Call prices should not be negative
assert np.all(call_Heston_Laplace >= -1e-8)

# 4. European call should not be worth more than the stock
assert np.all(call_Heston_Laplace <= S0 + 1e-8)

# 5. Call prices should be weakly decreasing in strike
assert np.all(np.diff(call_Heston_Laplace) <= 1e-8)

# 6. European call should respect the no-arbitrage lower bound
lower_bound = np.maximum(S0 - K_arr * np.exp(-r * tau), 0)
assert np.all(call_Heston_Laplace >= lower_bound - 1e-6)

# 7. Call prices should be convex in strike
assert np.all(np.diff(call_Heston_Laplace, n=2) >= -1e-6)


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Plot Heston call prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, call_Heston_Laplace, label="Heston call by Laplace inversion")
plt.plot(K_arr, lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Call values")
plt.title("Heston Call Price vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "heston_laplace_call_prices.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print(f"Minimum Heston call price: {call_Heston_Laplace.min():.6f}")
print(f"Maximum Heston call price: {call_Heston_Laplace.max():.6f}")
print(f"First strike: {K_arr[0]:.6f}")
print(f"Last strike: {K_arr[-1]:.6f}")
print(f"First call price: {call_Heston_Laplace[0]:.6f}")
print(f"Last call price: {call_Heston_Laplace[-1]:.6f}")
print(f"Plot saved to: {save_path}")
print("Heston Laplace call test passed.")



# --------------------------------------------------
# Compute Heston put prices using Laplace inversion
# --------------------------------------------------

R_put = -1

put_Heston_Laplace = eu_lap(
    heston_char,
    K_arr,
    R_put,
    S0,
    t,
    T,
    r,
    gam0,
    kappa,
    lamb,
    sig_tilde,
    rho,
)


# --------------------------------------------------
# Basic sanity checks for puts
# --------------------------------------------------

# 1. Output should have same shape as strike array
assert put_Heston_Laplace.shape == K_arr.shape

# 2. Prices should be finite
assert np.all(np.isfinite(put_Heston_Laplace))

# 3. Put prices should not be negative
assert np.all(put_Heston_Laplace >= -1e-8)

# 4. European put should not be worth more than discounted strike
assert np.all(put_Heston_Laplace <= K_arr * np.exp(-r * tau) + 1e-8)

# 5. Put prices should be weakly increasing in strike
assert np.all(np.diff(put_Heston_Laplace) >= -1e-8)

# 6. European put should respect the no-arbitrage lower bound
put_lower_bound = np.maximum(K_arr * np.exp(-r * tau) - S0, 0)
assert np.all(put_Heston_Laplace >= put_lower_bound - 1e-6)

# 7. Put prices should be convex in strike
assert np.all(np.diff(put_Heston_Laplace, n=2) >= -1e-6)

# 8. Put-call parity should hold approximately
parity_error = call_Heston_Laplace - put_Heston_Laplace - (
    S0 - K_arr * np.exp(-r * tau)
)

assert np.max(np.abs(parity_error)) < 1e-4


# --------------------------------------------------
# Plot Heston put prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, put_Heston_Laplace, label="Heston put by Laplace inversion")
plt.plot(K_arr, put_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Put values")
plt.title("Heston Put Price vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_put = os.path.join(plot_dir, "heston_laplace_put_prices.png")
plt.savefig(save_path_put, dpi=300, bbox_inches="tight")
plt.close()

print(f"Minimum Heston call price: {call_Heston_Laplace.min():.6f}")
print(f"Maximum Heston call price: {call_Heston_Laplace.max():.6f}")
print(f"Minimum Heston put price: {put_Heston_Laplace.min():.6f}")
print(f"Maximum Heston put price: {put_Heston_Laplace.max():.6f}")

print(f"First strike: {K_arr[0]:.6f}")
print(f"Last strike: {K_arr[-1]:.6f}")

print(f"First call price: {call_Heston_Laplace[0]:.6f}")
print(f"Last call price: {call_Heston_Laplace[-1]:.6f}")

print(f"First put price: {put_Heston_Laplace[0]:.6f}")
print(f"Last put price: {put_Heston_Laplace[-1]:.6f}")

print(f"Maximum put-call parity error: {np.max(np.abs(parity_error)):.8f}")

print(f"Call plot saved to: {save_path}")
print(f"Put plot saved to: {save_path_put}")

print("Heston Laplace call and put tests passed.")

# --------------------------------------------------
# Put-call parity check
# --------------------------------------------------

parity_lhs = call_Heston_Laplace - put_Heston_Laplace
parity_rhs = S0 - K_arr * np.exp(-r * tau)

parity_error = parity_lhs - parity_rhs
max_parity_error = np.max(np.abs(parity_error))

assert max_parity_error < 1e-4

print(f"Maximum put-call parity error: {max_parity_error:.8f}")

# --------------------------------------------------
# Black-Scholes parameters
# --------------------------------------------------

sigma = 0.3

R_call_BS = 2
R_put_BS = -1


# --------------------------------------------------
# Compute Black-Scholes call prices using Laplace inversion
# --------------------------------------------------

call_BS_Laplace = eu_lap(
    bs_char,
    K_arr,
    R_call_BS,
    S0,
    t,
    T,
    r,
    sigma,
)


# --------------------------------------------------
# Compute Black-Scholes put prices using Laplace inversion
# --------------------------------------------------

put_BS_Laplace = eu_lap(
    bs_char,
    K_arr,
    R_put_BS,
    S0,
    t,
    T,
    r,
    sigma,
)


# --------------------------------------------------
# No-arbitrage bounds
# --------------------------------------------------

bs_call_lower_bound = np.maximum(S0 - K_arr * np.exp(-r * tau), 0)
bs_put_lower_bound = np.maximum(K_arr * np.exp(-r * tau) - S0, 0)


# --------------------------------------------------
# Basic sanity checks for Black-Scholes calls
# --------------------------------------------------

assert call_BS_Laplace.shape == K_arr.shape
assert np.all(np.isfinite(call_BS_Laplace))
assert np.all(call_BS_Laplace >= -1e-8)
assert np.all(call_BS_Laplace <= S0 + 1e-8)
assert np.all(np.diff(call_BS_Laplace) <= 1e-8)
assert np.all(call_BS_Laplace >= bs_call_lower_bound - 1e-6)
assert np.all(np.diff(call_BS_Laplace, n=2) >= -1e-6)


# --------------------------------------------------
# Basic sanity checks for Black-Scholes puts
# --------------------------------------------------

assert put_BS_Laplace.shape == K_arr.shape
assert np.all(np.isfinite(put_BS_Laplace))
assert np.all(put_BS_Laplace >= -1e-8)
assert np.all(put_BS_Laplace <= K_arr * np.exp(-r * tau) + 1e-8)
assert np.all(np.diff(put_BS_Laplace) >= -1e-8)
assert np.all(put_BS_Laplace >= bs_put_lower_bound - 1e-6)
assert np.all(np.diff(put_BS_Laplace, n=2) >= -1e-6)


# --------------------------------------------------
# Black-Scholes put-call parity check
# --------------------------------------------------

bs_parity_lhs = call_BS_Laplace - put_BS_Laplace
bs_parity_rhs = S0 - K_arr * np.exp(-r * tau)

bs_parity_error = bs_parity_lhs - bs_parity_rhs
bs_max_parity_error = np.max(np.abs(bs_parity_error))

assert bs_max_parity_error < 1e-4


# --------------------------------------------------
# Plot Black-Scholes call prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, call_BS_Laplace, label="Black-Scholes call by Laplace inversion")
plt.plot(K_arr, bs_call_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Call values")
plt.title("Black-Scholes Call Price vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_call = os.path.join(plot_dir, "bs_laplace_call_prices.png")
plt.savefig(save_path_bs_call, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Black-Scholes put prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, put_BS_Laplace, label="Black-Scholes put by Laplace inversion")
plt.plot(K_arr, bs_put_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Put values")
plt.title("Black-Scholes Put Price vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_put = os.path.join(plot_dir, "bs_laplace_put_prices.png")
plt.savefig(save_path_bs_put, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Black-Scholes put-call parity error
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, bs_parity_error, label="Put-call parity error")
plt.axhline(0, linestyle="--")

plt.xlabel("Strike values")
plt.ylabel("Parity error")
plt.title("Put-Call Parity Error for Black-Scholes Laplace Prices")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_parity = os.path.join(plot_dir, "bs_laplace_put_call_parity_error.png")
plt.savefig(save_path_bs_parity, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Black-Scholes diagnostics
# --------------------------------------------------

print(f"Minimum Black-Scholes Laplace call price: {call_BS_Laplace.min():.6f}")
print(f"Maximum Black-Scholes Laplace call price: {call_BS_Laplace.max():.6f}")

print(f"Minimum Black-Scholes Laplace put price: {put_BS_Laplace.min():.6f}")
print(f"Maximum Black-Scholes Laplace put price: {put_BS_Laplace.max():.6f}")

print(f"First Black-Scholes call price: {call_BS_Laplace[0]:.6f}")
print(f"Last Black-Scholes call price: {call_BS_Laplace[-1]:.6f}")

print(f"First Black-Scholes put price: {put_BS_Laplace[0]:.6f}")
print(f"Last Black-Scholes put price: {put_BS_Laplace[-1]:.6f}")

print(f"Maximum Black-Scholes put-call parity error: {bs_max_parity_error:.8f}")

print(f"Black-Scholes call plot saved to: {save_path_bs_call}")
print(f"Black-Scholes put plot saved to: {save_path_bs_put}")
print(f"Black-Scholes parity error plot saved to: {save_path_bs_parity}")

print("Black-Scholes Laplace call and put tests passed.")