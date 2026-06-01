import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.american import perp_am_put
from comp_fin_lab.bs import eu_bs
from comp_fin_lab.payoffs import vcall, vput
from comp_fin_lab.binom import eu_crr
from comp_fin_lab.american import am_crr

# --------------------------------------------------
# Parameters
# --------------------------------------------------

K = 100
r = 0.02
sigma = 0.3

S_min = 10**-6
S_max = 200
n_grid = 200

# European put maturity used only for comparison
T_eu = 1


# --------------------------------------------------
# Compute perpetual American put values
# --------------------------------------------------

S_grid, v_grid_am, x_star = perp_am_put(
    K=K,
    r=r,
    sigma=sigma,
    S_min=S_min,
    S_max=S_max,
    n_grid=n_grid,
)


# --------------------------------------------------
# Compute European put values for comparison
# --------------------------------------------------

v_grid_eu = eu_bs(
    t=0,
    St=S_grid,
    K=K,
    T=T_eu,
    r=r,
    sigma=sigma,
    call=0,
)


# --------------------------------------------------
# Basic checks
# --------------------------------------------------

payoff = np.maximum(K - S_grid, 0.0)

print(f"Exercise boundary x_star: {x_star:.6f}")
print(f"American put value at lowest S: {v_grid_am[0]:.6f}")
print(f"American put value at highest S: {v_grid_am[-1]:.6f}")
print(f"European put value at highest S: {v_grid_eu[-1]:.6f}")

# All values should be finite
assert np.all(np.isfinite(v_grid_am))
assert np.all(np.isfinite(v_grid_eu))

# Perpetual American put values should be non-negative
assert np.all(v_grid_am >= -1e-10)

# Exercise boundary should be between 0 and K for this put case
assert 0 < x_star < K

# In the exercise region, value should equal immediate payoff
exercise_mask = S_grid <= x_star
assert np.allclose(v_grid_am[exercise_mask], payoff[exercise_mask], atol=1e-8)

# American option value should be at least intrinsic value
assert np.all(v_grid_am >= payoff - 1e-8)

# Perpetual American put should be at least as valuable as European put with finite maturity
assert np.all(v_grid_am >= v_grid_eu - 1e-6)

# Put value should generally decrease as stock price increases
assert np.all(np.diff(v_grid_am) <= 1e-6)


# --------------------------------------------------
# Save plot
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(S_grid, v_grid_am, label="Perpetual American put")
plt.plot(S_grid, v_grid_eu, label="European put, T = 1")
plt.plot(S_grid, payoff, linestyle="--", label="Immediate exercise payoff")
plt.axvline(x_star, linestyle="--", label=r"$x^*$")

plt.xlabel("Stock price S")
plt.ylabel("Put value")
plt.title("Perpetual American Put vs European Put")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "perpetual_american_put.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to: {save_path}")
print("Perpetual American put test passed.")


# --------------------------------------------------
# Parameters
# --------------------------------------------------

S0 = 120
K = 100
T = 1
r = 0.02
sigma = 0.3
N = 500
c = 1


# --------------------------------------------------
# 1. Compare American CRR and European CRR prices
# --------------------------------------------------

eu_call = eu_crr(
    g=vcall,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=N,
    c=c,
)

eu_put = eu_crr(
    g=vput,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=N,
    c=c,
)

am_call, C_call, S_call = am_crr(
    S_ini=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=N,
    opttype="C",
    c=c,
)

am_put, C_put, S_put = am_crr(
    S_ini=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    N=N,
    opttype="P",
    c=c,
)

print("\nAmerican CRR vs European CRR")
print(f"European call CRR: {eu_call:.8f}")
print(f"American call CRR: {am_call:.8f}")
print(f"European put CRR:  {eu_put:.8f}")
print(f"American put CRR:  {am_put:.8f}")

# Basic finite checks
assert np.isfinite(eu_call)
assert np.isfinite(eu_put)
assert np.isfinite(am_call)
assert np.isfinite(am_put)

# American option should never be worth less than European option
assert am_call >= eu_call - 1e-8
assert am_put >= eu_put - 1e-8

# For non-dividend-paying stocks, American call should be essentially equal to European call
assert abs(am_call - eu_call) < 1e-6

# American put can be more valuable because early exercise may be optimal
assert am_put >= eu_put - 1e-8

# Matrices should have expected shape
assert C_call.shape == (N + 1, N + 1)
assert S_call.shape == (N + 1, N + 1)
assert C_put.shape == (N + 1, N + 1)
assert S_put.shape == (N + 1, N + 1)


# --------------------------------------------------
# 2. Convergence test for increasing N
# --------------------------------------------------

N_values = np.array([5, 10, 25, 50, 100, 200, 500])

am_call_prices = np.zeros_like(N_values, dtype=float)
eu_call_prices = np.zeros_like(N_values, dtype=float)
am_put_prices = np.zeros_like(N_values, dtype=float)
eu_put_prices = np.zeros_like(N_values, dtype=float)

for i, n in enumerate(N_values):
    eu_call_prices[i] = eu_crr(
        g=vcall,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=int(n),
        c=c,
    )

    eu_put_prices[i] = eu_crr(
        g=vput,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        M=int(n),
        c=c,
    )

    am_call_prices[i], _, _ = am_crr(
        S_ini=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        N=int(n),
        opttype="C",
        c=c,
    )

    am_put_prices[i], _, _ = am_crr(
        S_ini=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        N=int(n),
        opttype="P",
        c=c,
    )

print("\nConvergence over N")
for n, ac, ec, ap, ep in zip(
    N_values,
    am_call_prices,
    eu_call_prices,
    am_put_prices,
    eu_put_prices,
):
    print(
        f"N={n:4d} | "
        f"AM call={ac:.8f}, EU call={ec:.8f}, "
        f"AM put={ap:.8f}, EU put={ep:.8f}"
    )

# American call and European call should be close for all N
assert np.all(am_call_prices >= eu_call_prices - 1e-8)
assert np.all(np.abs(am_call_prices - eu_call_prices) < 1e-6)

# American put should dominate European put for all N
assert np.all(am_put_prices >= eu_put_prices - 1e-8)

# Last few prices should be reasonably stable
assert abs(am_call_prices[-1] - am_call_prices[-2]) < 0.5
assert abs(am_put_prices[-1] - am_put_prices[-2]) < 0.5


# --------------------------------------------------
# Save convergence plot
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(9, 5))

plt.plot(N_values, am_call_prices, marker="o", label="American call")
plt.plot(N_values, eu_call_prices, marker="o", label="European call")
plt.plot(N_values, am_put_prices, marker="o", label="American put")
plt.plot(N_values, eu_put_prices, marker="o", label="European put")

plt.xlabel("Number of time steps N")
plt.ylabel("Option price")
plt.title("American and European CRR Prices as N Increases")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "american_crr_convergence.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Convergence plot saved to: {save_path}")


# --------------------------------------------------
# 3. Volatility sensitivity test
# --------------------------------------------------

sigmas = np.linspace(0.05, 1.0, 100)

am_call_sigma = np.zeros_like(sigmas)
eu_call_sigma = np.zeros_like(sigmas)
am_put_sigma = np.zeros_like(sigmas)
eu_put_sigma = np.zeros_like(sigmas)

for i, sig in enumerate(sigmas):
    eu_call_sigma[i] = eu_crr(
        g=vcall,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sig,
        M=N,
        c=c,
    )

    eu_put_sigma[i] = eu_crr(
        g=vput,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sig,
        M=N,
        c=c,
    )

    am_call_sigma[i], _, _ = am_crr(
        S_ini=S0,
        K=K,
        T=T,
        r=r,
        sigma=sig,
        N=N,
        opttype="C",
        c=c,
    )

    am_put_sigma[i], _, _ = am_crr(
        S_ini=S0,
        K=K,
        T=T,
        r=r,
        sigma=sig,
        N=N,
        opttype="P",
        c=c,
    )

# Basic checks over sigma
assert np.all(np.isfinite(am_call_sigma))
assert np.all(np.isfinite(eu_call_sigma))
assert np.all(np.isfinite(am_put_sigma))
assert np.all(np.isfinite(eu_put_sigma))

assert np.all(am_call_sigma >= -1e-10)
assert np.all(eu_call_sigma >= -1e-10)
assert np.all(am_put_sigma >= -1e-10)
assert np.all(eu_put_sigma >= -1e-10)

# American dominates European
assert np.all(am_call_sigma >= eu_call_sigma - 1e-8)
assert np.all(am_put_sigma >= eu_put_sigma - 1e-8)

# Non-dividend American call should be approximately equal to European call
assert np.max(np.abs(am_call_sigma - eu_call_sigma)) < 1e-6

# Option prices should generally increase with volatility
assert np.all(np.diff(am_call_sigma) >= -1e-6)
assert np.all(np.diff(eu_call_sigma) >= -1e-6)
assert np.all(np.diff(am_put_sigma) >= -1e-6)
assert np.all(np.diff(eu_put_sigma) >= -1e-6)


# --------------------------------------------------
# Save volatility sensitivity plot
# --------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(sigmas, am_call_sigma, label="American call")
ax[0].plot(sigmas, eu_call_sigma, linestyle="--", label="European call")
ax[0].set_xlabel(r"$\sigma$")
ax[0].set_ylabel("Call price")
ax[0].set_title("Call Prices vs Volatility")
ax[0].grid(alpha=0.3)
ax[0].legend()

ax[1].plot(sigmas, am_put_sigma, label="American put")
ax[1].plot(sigmas, eu_put_sigma, linestyle="--", label="European put")
ax[1].set_xlabel(r"$\sigma$")
ax[1].set_ylabel("Put price")
ax[1].set_title("Put Prices vs Volatility")
ax[1].grid(alpha=0.3)
ax[1].legend()

fig.suptitle("American vs European CRR Prices under Volatility Variation")
fig.tight_layout()

save_path = os.path.join(plot_dir, "american_crr_sigma_sensitivity.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Volatility sensitivity plot saved to: {save_path}")


# --------------------------------------------------
# 4. Early exercise premium plot
# --------------------------------------------------

call_premium = am_call_sigma - eu_call_sigma
put_premium = am_put_sigma - eu_put_sigma

plt.figure(figsize=(8, 5))

plt.plot(sigmas, call_premium, label="Call early-exercise premium")
plt.plot(sigmas, put_premium, label="Put early-exercise premium")
plt.axhline(0, linestyle="--")

plt.xlabel(r"$\sigma$")
plt.ylabel("American price - European price")
plt.title("Early Exercise Premium vs Volatility")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "american_crr_early_exercise_premium.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Early exercise premium plot saved to: {save_path}")

print("American CRR tests passed.")