import os
import time

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.payoffs import vcall, pcall
from comp_fin_lab.bs import eu_bs
from comp_fin_lab.binom import eu_crr
from comp_fin_lab.integration import eu_int_sn


# --------------------------------------------------
# Parameters
# --------------------------------------------------

r = 0.02
sigma = 0.3
T = 1
t = 0
K = 100
S0 = 120
M = 1000
alpha = 1.2


# --------------------------------------------------
# 1. Check integration price against Black-Scholes
# --------------------------------------------------
# eu_int_sn expects f(S), not f(S, K), so we wrap vcall with lambda.

call_price_int = eu_int_sn(
    t=t,
    f=lambda S: vcall(S, K),
    St=S0,
    T=T,
    r=r,
    sigma=sigma,
    a=-np.inf,
    b=np.inf
)

reference_price_BS = eu_bs(
    t=t,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
)

relative_abs_error = np.abs(reference_price_BS - call_price_int) / reference_price_BS

print("European call by integration:", call_price_int)
print("European call by Black-Scholes:", reference_price_BS)
print("Relative absolute error:", relative_abs_error)

assert np.isfinite(call_price_int)
assert np.isfinite(reference_price_BS)
assert relative_abs_error < 1e-6


# --------------------------------------------------
# 2. Compare power-call pricing by integration and CRR
# --------------------------------------------------
# For integration: f(S)
# For CRR: g(S, K)

power_call_price_int = eu_int_sn(
    t=t,
    f=lambda S: pcall(S, K, alpha),
    St=S0,
    T=T,
    r=r,
    sigma=sigma,
    a=-np.inf,
    b=np.inf,
)

power_call_price_crr = eu_crr(
    g=lambda S, K: pcall(S, K, alpha),
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
    c=1,
)

relative_power_call_error = (
    np.abs(power_call_price_crr - power_call_price_int) / np.abs(power_call_price_int)
)

print("\nPower call by integration:", power_call_price_int)
print("Power call by CRR:", power_call_price_crr)
print("Relative difference between CRR and integration:", relative_power_call_error)

assert np.isfinite(power_call_price_int)
assert np.isfinite(power_call_price_crr)
assert power_call_price_int >= 0
assert power_call_price_crr >= 0

# CRR is an approximation, so allow a looser tolerance here.
assert relative_power_call_error < 0.05


# --------------------------------------------------
# 3. Timing comparison: integration vs CRR
# --------------------------------------------------

n_runs = 20

start = time.perf_counter()

for _ in range(n_runs):
    eu_int_sn(
        t=t,
        f=lambda S: pcall(S, K, alpha),
        St=S0,
        T=T,
        r=r,
        sigma=sigma,
        a=-np.inf,
        b=np.inf
    )

integration_time = time.perf_counter() - start


start = time.perf_counter()

for _ in range(n_runs):
    eu_crr(
        g=lambda S, K: pcall(S, K, alpha),
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=M,
        c=1,
    )

crr_time = time.perf_counter() - start

integration_avg_time = integration_time / n_runs
crr_avg_time = crr_time / n_runs
speedup = crr_avg_time / integration_avg_time

print("\nTiming comparison")
print(f"Average integration time: {integration_avg_time:.8f} seconds")
print(f"Average CRR time:         {crr_avg_time:.8f} seconds")
print(f"Integration speedup over CRR: {speedup:.2f}x")

assert integration_avg_time < crr_avg_time
assert speedup > 2


# --------------------------------------------------
# 4. Save timing plot
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(7, 5))

plt.bar(
    ["Integration", "CRR"],
    [integration_avg_time, crr_avg_time],
)

plt.ylabel("Average runtime in seconds")
plt.title("Power Call Pricing Runtime: Integration vs CRR")
plt.grid(axis="y", alpha=0.3)

save_path = os.path.join(plot_dir, "integration_vs_crr_power_call_runtime.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nPlot saved to: {save_path}")
print("Integration test passed.")