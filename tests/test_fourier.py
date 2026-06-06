import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.char_fun import heston_char, bs_char
from comp_fin_lab.fourier import eu_fourier
from timeit import repeat
from comp_fin_lab.laplace import eu_lap


# --------------------------------------------------
# Parameters
# --------------------------------------------------

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
tau = T - t

M = 5000
N = 2**15


# --------------------------------------------------
# Compute Heston call prices using Fourier / FFT
# --------------------------------------------------

R_call = 2

call_Heston_Fourier = eu_fourier(
    heston_char,
    K_arr,
    R_call,
    S0,
    t,
    T,
    r,
    M,
    N,
    gam0,
    kappa,
    lamb,
    sig_tilde,
    rho,
)


# --------------------------------------------------
# Compute Heston put prices using Fourier / FFT
# --------------------------------------------------

R_put = -1

put_Heston_Fourier = eu_fourier(
    heston_char,
    K_arr,
    R_put,
    S0,
    t,
    T,
    r,
    M,
    N,
    gam0,
    kappa,
    lamb,
    sig_tilde,
    rho,
)


# --------------------------------------------------
# No-arbitrage bounds
# --------------------------------------------------

call_lower_bound = np.maximum(S0 - K_arr * np.exp(-r * tau), 0)
put_lower_bound = np.maximum(K_arr * np.exp(-r * tau) - S0, 0)


# --------------------------------------------------
# Basic sanity checks for calls
# --------------------------------------------------

# 1. Output should have same shape as strike array
assert call_Heston_Fourier.shape == K_arr.shape

# 2. Prices should be finite
assert np.all(np.isfinite(call_Heston_Fourier))

# 3. Call prices should not be negative
assert np.all(call_Heston_Fourier >= -1e-6)

# 4. European call should not be worth more than the stock
assert np.all(call_Heston_Fourier <= S0 + 1e-6)

# 5. Call prices should be weakly decreasing in strike
assert np.all(np.diff(call_Heston_Fourier) <= 1e-5)

# 6. European call should respect the no-arbitrage lower bound
assert np.all(call_Heston_Fourier >= call_lower_bound - 1e-4)

# 7. Call prices should be convex in strike
assert np.all(np.diff(call_Heston_Fourier, n=2) >= -1e-4)


# --------------------------------------------------
# Basic sanity checks for puts
# --------------------------------------------------

# 1. Output should have same shape as strike array
assert put_Heston_Fourier.shape == K_arr.shape

# 2. Prices should be finite
assert np.all(np.isfinite(put_Heston_Fourier))

# 3. Put prices should not be negative
assert np.all(put_Heston_Fourier >= -1e-6)

# 4. European put should not be worth more than discounted strike
assert np.all(put_Heston_Fourier <= K_arr * np.exp(-r * tau) + 1e-6)

# 5. Put prices should be weakly increasing in strike
assert np.all(np.diff(put_Heston_Fourier) >= -1e-5)

# 6. European put should respect the no-arbitrage lower bound
assert np.all(put_Heston_Fourier >= put_lower_bound - 1e-4)

# 7. Put prices should be convex in strike
assert np.all(np.diff(put_Heston_Fourier, n=2) >= -1e-4)


# --------------------------------------------------
# Put-call parity check
# --------------------------------------------------

parity_lhs = call_Heston_Fourier - put_Heston_Fourier
parity_rhs = S0 - K_arr * np.exp(-r * tau)

parity_error = parity_lhs - parity_rhs
max_parity_error = np.max(np.abs(parity_error))

assert max_parity_error < 1e-3


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Plot Heston call prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, call_Heston_Fourier, label="Heston call by Fourier / FFT")
plt.plot(K_arr, call_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Call values")
plt.title("Heston Call Price vs Strike using Fourier / FFT")
plt.grid(alpha=0.3)
plt.legend()

save_path_call = os.path.join(plot_dir, "heston_fourier_call_prices.png")
plt.savefig(save_path_call, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Heston put prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, put_Heston_Fourier, label="Heston put by Fourier / FFT")
plt.plot(K_arr, put_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Put values")
plt.title("Heston Put Price vs Strike using Fourier / FFT")
plt.grid(alpha=0.3)
plt.legend()

save_path_put = os.path.join(plot_dir, "heston_fourier_put_prices.png")
plt.savefig(save_path_put, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot put-call parity error
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, parity_error, label="Put-call parity error")
plt.axhline(0, linestyle="--")

plt.xlabel("Strike values")
plt.ylabel("Parity error")
plt.title("Put-Call Parity Error for Heston Fourier / FFT Prices")
plt.grid(alpha=0.3)
plt.legend()

save_path_parity = os.path.join(plot_dir, "heston_fourier_put_call_parity_error.png")
plt.savefig(save_path_parity, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print(f"Minimum Heston Fourier call price: {call_Heston_Fourier.min():.6f}")
print(f"Maximum Heston Fourier call price: {call_Heston_Fourier.max():.6f}")

print(f"Minimum Heston Fourier put price: {put_Heston_Fourier.min():.6f}")
print(f"Maximum Heston Fourier put price: {put_Heston_Fourier.max():.6f}")

print(f"First strike: {K_arr[0]:.6f}")
print(f"Last strike: {K_arr[-1]:.6f}")

print(f"First call price: {call_Heston_Fourier[0]:.6f}")
print(f"Last call price: {call_Heston_Fourier[-1]:.6f}")

print(f"First put price: {put_Heston_Fourier[0]:.6f}")
print(f"Last put price: {put_Heston_Fourier[-1]:.6f}")

print(f"Maximum put-call parity error: {max_parity_error:.8f}")

print(f"Call plot saved to: {save_path_call}")
print(f"Put plot saved to: {save_path_put}")
print(f"Parity error plot saved to: {save_path_parity}")

print("Heston Fourier call and put tests passed.")


# --------------------------------------------------
# Timing comparison: Laplace quadrature vs Fourier / FFT
# --------------------------------------------------

def run_laplace_call():
    return eu_lap(
        heston_char,
        K_arr,
        R_call,
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


def run_fourier_call():
    return eu_fourier(
        heston_char,
        K_arr,
        R_call,
        S0,
        t,
        T,
        r,
        M,
        N,
        gam0,
        kappa,
        lamb,
        sig_tilde,
        rho,
    )


# Small warm-up run
_ = run_fourier_call()

laplace_times = repeat(run_laplace_call, repeat=3, number=1)
fourier_times = repeat(run_fourier_call, repeat=3, number=1)

laplace_time = min(laplace_times)
fourier_time = min(fourier_times)

speedup = laplace_time / fourier_time

assert fourier_time < laplace_time

print(f"Fastest Laplace runtime: {laplace_time:.6f} seconds")
print(f"Fastest Fourier runtime: {fourier_time:.6f} seconds")
print(f"Fourier speedup factor: {speedup:.2f}x")

# --------------------------------------------------
# Black-Scholes parameters
# --------------------------------------------------

sigma = 0.3

R_call_BS = 2
R_put_BS = -1


# --------------------------------------------------
# Compute Black-Scholes call prices using Fourier / FFT
# --------------------------------------------------

call_BS_Fourier = eu_fourier(
    bs_char,
    K_arr,
    R_call_BS,
    S0,
    t,
    T,
    r,
    M,
    N,
    sigma,
)


# --------------------------------------------------
# Compute Black-Scholes put prices using Fourier / FFT
# --------------------------------------------------

put_BS_Fourier = eu_fourier(
    bs_char,
    K_arr,
    R_put_BS,
    S0,
    t,
    T,
    r,
    M,
    N,
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

assert call_BS_Fourier.shape == K_arr.shape
assert np.all(np.isfinite(call_BS_Fourier))
assert np.all(call_BS_Fourier >= -1e-6)
assert np.all(call_BS_Fourier <= S0 + 1e-6)
assert np.all(np.diff(call_BS_Fourier) <= 1e-5)
assert np.all(call_BS_Fourier >= bs_call_lower_bound - 1e-4)
assert np.all(np.diff(call_BS_Fourier, n=2) >= -1e-4)


# --------------------------------------------------
# Basic sanity checks for Black-Scholes puts
# --------------------------------------------------

assert put_BS_Fourier.shape == K_arr.shape
assert np.all(np.isfinite(put_BS_Fourier))
assert np.all(put_BS_Fourier >= -1e-6)
assert np.all(put_BS_Fourier <= K_arr * np.exp(-r * tau) + 1e-6)
assert np.all(np.diff(put_BS_Fourier) >= -1e-5)
assert np.all(put_BS_Fourier >= bs_put_lower_bound - 1e-4)
assert np.all(np.diff(put_BS_Fourier, n=2) >= -1e-4)


# --------------------------------------------------
# Black-Scholes put-call parity check
# --------------------------------------------------

bs_parity_lhs = call_BS_Fourier - put_BS_Fourier
bs_parity_rhs = S0 - K_arr * np.exp(-r * tau)

bs_parity_error = bs_parity_lhs - bs_parity_rhs
bs_max_parity_error = np.max(np.abs(bs_parity_error))

assert bs_max_parity_error < 1e-3


# --------------------------------------------------
# Plot Black-Scholes call prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, call_BS_Fourier, label="Black-Scholes call by Fourier / FFT")
plt.plot(K_arr, bs_call_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Call values")
plt.title("Black-Scholes Call Price vs Strike using Fourier / FFT")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_call = os.path.join(plot_dir, "bs_fourier_call_prices.png")
plt.savefig(save_path_bs_call, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Black-Scholes put prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, put_BS_Fourier, label="Black-Scholes put by Fourier / FFT")
plt.plot(K_arr, bs_put_lower_bound, linestyle="--", label="No-arbitrage lower bound")

plt.xlabel("Strike values")
plt.ylabel("Put values")
plt.title("Black-Scholes Put Price vs Strike using Fourier / FFT")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_put = os.path.join(plot_dir, "bs_fourier_put_prices.png")
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
plt.title("Put-Call Parity Error for Black-Scholes Fourier / FFT Prices")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_parity = os.path.join(plot_dir, "bs_fourier_put_call_parity_error.png")
plt.savefig(save_path_bs_parity, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Black-Scholes diagnostics
# --------------------------------------------------

print(f"Minimum Black-Scholes Fourier call price: {call_BS_Fourier.min():.6f}")
print(f"Maximum Black-Scholes Fourier call price: {call_BS_Fourier.max():.6f}")

print(f"Minimum Black-Scholes Fourier put price: {put_BS_Fourier.min():.6f}")
print(f"Maximum Black-Scholes Fourier put price: {put_BS_Fourier.max():.6f}")

print(f"First Black-Scholes call price: {call_BS_Fourier[0]:.6f}")
print(f"Last Black-Scholes call price: {call_BS_Fourier[-1]:.6f}")

print(f"First Black-Scholes put price: {put_BS_Fourier[0]:.6f}")
print(f"Last Black-Scholes put price: {put_BS_Fourier[-1]:.6f}")

print(f"Maximum Black-Scholes put-call parity error: {bs_max_parity_error:.8f}")

print(f"Black-Scholes call plot saved to: {save_path_bs_call}")
print(f"Black-Scholes put plot saved to: {save_path_bs_put}")
print(f"Black-Scholes parity error plot saved to: {save_path_bs_parity}")

print("Black-Scholes Fourier call and put tests passed.")