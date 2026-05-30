import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.payoffs import vcall
from comp_fin_lab.binom import eu_crr
from comp_fin_lab.barriers import down_and_out_crr


# --------------------------------------------------
# Parameters
# --------------------------------------------------

r = 0.02
sigma = 0.3
T = 1
K = 100
S0 = 120
M = 500
c = 1

B_values = np.linspace(50, 150, 100)


# --------------------------------------------------
# Compute down-and-out call prices for different barriers
# --------------------------------------------------

barrier_prices_CRR = np.zeros_like(B_values)

for i in range(len(B_values)):
    barrier_prices_CRR[i] = down_and_out_crr(
        g=vcall,
        S0=S0,
        K=K,
        T=T,
        B=B_values[i],
        r=r,
        sigma=sigma,
        M=M,
        c=c,
    )


# --------------------------------------------------
# Compute vanilla CRR call price for comparison
# --------------------------------------------------

vanilla_call_price = eu_crr(
    g=vcall,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
    c=c,
)


# --------------------------------------------------
# Basic sanity checks
# --------------------------------------------------

# 1. Barrier prices should not be negative
assert np.all(barrier_prices_CRR >= -1e-10)

# 2. A down-and-out call should not be worth more than the vanilla call
assert np.all(barrier_prices_CRR <= vanilla_call_price + 1e-10)

# 3. As the barrier increases, the option should become weakly cheaper
assert np.all(np.diff(barrier_prices_CRR) <= 1e-8)

# 4. If the barrier is above S0, the option should be knocked out immediately
assert np.all(barrier_prices_CRR[B_values > S0] == 0)


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Plot barrier prices
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(B_values, barrier_prices_CRR, label="Down-and-out call")
plt.axhline(vanilla_call_price, linestyle="--", label="Vanilla CRR call")
plt.axvline(S0, linestyle="--", label="$S_0$")

plt.xlabel("Barrier values")
plt.ylabel("Call values")
plt.title("Down-and-Out Call Price vs Barrier Level")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "down_and_out_crr_barrier_sensitivity.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Vanilla call price: {vanilla_call_price:.6f}")
print(f"Minimum barrier price: {barrier_prices_CRR.min():.6f}")
print(f"Maximum barrier price: {barrier_prices_CRR.max():.6f}")
print(f"Plot saved to: {save_path}")
print("Barrier test passed.")