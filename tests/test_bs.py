import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.payoffs import vcall
from comp_fin_lab.binom import eu_crr
from comp_fin_lab.bs import eu_bs


# --------------------------------------------------
# Parameters
# --------------------------------------------------

r = 0.02
sigma = 0.3
T = 1
K = 100
t = 0
M = 100

S_arr = np.linspace(20, 300, 1000)


# --------------------------------------------------
# Compute CRR prices, anchored CRR prices, and BS prices
# --------------------------------------------------

call_prices_CRR = np.zeros_like(S_arr)
call_prices_CRR_anchored = np.zeros_like(S_arr)
call_prices_BS = np.zeros_like(S_arr)

for i in range(len(S_arr)):
    S0 = S_arr[i]

    call_prices_CRR[i] = eu_crr(
        g=vcall,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M,
        c=1,
    )

    call_prices_CRR_anchored[i] = eu_crr(
        g=vcall,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M,
        c=(K / S0) ** (2 / M),
    )

    call_prices_BS[i] = eu_bs(
        t=t,
        St=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        call=1,
    )


# --------------------------------------------------
# Compute pricing errors against Black-Scholes
# --------------------------------------------------

crr_error = call_prices_CRR - call_prices_BS
crr_anchored_error = call_prices_CRR_anchored - call_prices_BS


# --------------------------------------------------
# Basic sanity checks
# --------------------------------------------------

assert np.all(np.isfinite(call_prices_CRR))
assert np.all(np.isfinite(call_prices_CRR_anchored))
assert np.all(np.isfinite(call_prices_BS))

assert np.all(call_prices_CRR >= -1e-10)
assert np.all(call_prices_CRR_anchored >= -1e-10)
assert np.all(call_prices_BS >= -1e-10)


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Plot CRR approximation errors against Black-Scholes
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(S_arr, crr_error, label="CRR - BS")
plt.plot(S_arr, crr_anchored_error, label="CRR anchored - BS")
plt.axhline(0, linestyle="--")

plt.xlabel("$S(0)$")
plt.ylabel("Difference between CRR and BS call prices")
plt.title("CRR vs Anchored CRR Approximation Error")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "bs_crr_anchored_error.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Print diagnostics
# --------------------------------------------------

print(f"Maximum absolute CRR error: {np.max(np.abs(crr_error)):.8f}")
print(f"Maximum absolute anchored CRR error: {np.max(np.abs(crr_anchored_error)):.8f}")

print(f"Mean absolute CRR error: {np.mean(np.abs(crr_error)):.8f}")
print(f"Mean absolute anchored CRR error: {np.mean(np.abs(crr_anchored_error)):.8f}")

print(f"Plot saved to: {save_path}")
print("Black-Scholes / CRR comparison test passed.")