import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.bs import eu_bs, bs_pde


# --------------------------------------------------
# Parameters
# --------------------------------------------------

S0 = 120
K = 100
T = 1
r = 0.02
sigma = 0.3

S_max = max(4 * K, 4 * S0)
S_steps = 300
t_steps = 20000


# --------------------------------------------------
# 1. Single-price comparison: PDE vs Black-Scholes
# --------------------------------------------------

call_pde, V_call, S_grid, tau_grid = bs_pde(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
    S_max=S_max,
    S_steps=S_steps,
    t_steps=t_steps,
)

put_pde, V_put, _, _ = bs_pde(
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=0,
    S_max=S_max,
    S_steps=S_steps,
    t_steps=t_steps,
)

call_bs = eu_bs(
    t=0,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
)

put_bs = eu_bs(
    t=0,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=0,
)

call_error = abs(call_pde - call_bs)
put_error = abs(put_pde - put_bs)

print("Single-price PDE vs Black-Scholes")
print(f"PDE call price:          {call_pde:.8f}")
print(f"Black-Scholes call price:{call_bs:.8f}")
print(f"Call absolute error:     {call_error:.8f}")

print(f"\nPDE put price:           {put_pde:.8f}")
print(f"Black-Scholes put price: {put_bs:.8f}")
print(f"Put absolute error:      {put_error:.8f}")

assert np.isfinite(call_pde)
assert np.isfinite(put_pde)
assert np.isfinite(call_bs)
assert np.isfinite(put_bs)

assert call_pde >= -1e-10
assert put_pde >= -1e-10

# Explicit PDE is an approximation, so allow a moderate tolerance.
assert call_error < 0.25
assert put_error < 0.25


# --------------------------------------------------
# 2. Grid and boundary checks
# --------------------------------------------------

assert V_call.shape == (t_steps + 1, S_steps + 1)
assert V_put.shape == (t_steps + 1, S_steps + 1)
assert S_grid.shape == (S_steps + 1,)
assert tau_grid.shape == (t_steps + 1,)

assert np.all(np.isfinite(V_call))
assert np.all(np.isfinite(V_put))

# Initial condition at tau = 0 equals payoff
assert np.allclose(V_call[0], np.maximum(S_grid - K, 0.0))
assert np.allclose(V_put[0], np.maximum(K - S_grid, 0.0))

# Boundary conditions
assert np.allclose(V_call[:, 0], 0.0)
assert np.allclose(V_put[:, -1], 0.0)

assert np.allclose(
    V_call[:, -1],
    S_max - K * np.exp(-r * tau_grid),
)

assert np.allclose(
    V_put[:, 0],
    K * np.exp(-r * tau_grid),
)


# --------------------------------------------------
# 3. Price curve comparison against Black-Scholes
# --------------------------------------------------

# Avoid S = 0 because Black-Scholes closed form contains log(S/K).
interior_mask = (S_grid > 1e-8) & (S_grid < S_max)

S_eval = S_grid[interior_mask]

call_pde_curve = V_call[-1, interior_mask]
put_pde_curve = V_put[-1, interior_mask]

call_bs_curve = eu_bs(
    t=0,
    St=S_eval,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
)

put_bs_curve = eu_bs(
    t=0,
    St=S_eval,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=0,
)

call_abs_error_curve = np.abs(call_pde_curve - call_bs_curve)
put_abs_error_curve = np.abs(put_pde_curve - put_bs_curve)

# Focus on the economically relevant central region.
central_mask = (S_eval >= 0.25 * K) & (S_eval <= 3.0 * K)

print("\nCurve diagnostics")
print(f"Mean call absolute error, central region: {call_abs_error_curve[central_mask].mean():.8f}")
print(f"Mean put absolute error, central region:  {put_abs_error_curve[central_mask].mean():.8f}")

assert call_abs_error_curve[central_mask].mean() < 0.20
assert put_abs_error_curve[central_mask].mean() < 0.20


# --------------------------------------------------
# Create plots folder inside tests
# --------------------------------------------------

test_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(test_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# 4. Plot PDE and Black-Scholes price curves
# --------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(S_eval, call_pde_curve, label="PDE call")
ax[0].plot(S_eval, call_bs_curve, linestyle="--", label="Black-Scholes call")
ax[0].set_xlabel("Stock price S")
ax[0].set_ylabel("Call price")
ax[0].set_title("Call: PDE vs Black-Scholes")
ax[0].grid(alpha=0.3)
ax[0].legend()

ax[1].plot(S_eval, put_pde_curve, label="PDE put")
ax[1].plot(S_eval, put_bs_curve, linestyle="--", label="Black-Scholes put")
ax[1].set_xlabel("Stock price S")
ax[1].set_ylabel("Put price")
ax[1].set_title("Put: PDE vs Black-Scholes")
ax[1].grid(alpha=0.3)
ax[1].legend()

fig.suptitle("Black-Scholes PDE Finite-Difference Prices")
fig.tight_layout()

save_path = os.path.join(plot_dir, "bs_pde_vs_black_scholes_prices.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Price curve plot saved to: {save_path}")


# --------------------------------------------------
# 5. Plot absolute pricing errors
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(S_eval, call_abs_error_curve, label="Call absolute error")
plt.plot(S_eval, put_abs_error_curve, label="Put absolute error")

plt.xlabel("Stock price S")
plt.ylabel("Absolute pricing error")
plt.title("Black-Scholes PDE Absolute Pricing Error")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "bs_pde_absolute_pricing_error.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Error plot saved to: {save_path}")


# --------------------------------------------------
# 6. Convergence test as the stock grid becomes finer
# --------------------------------------------------

S_steps_values = np.array([50, 100, 200, 300])

call_errors = np.zeros_like(S_steps_values, dtype=float)
put_errors = np.zeros_like(S_steps_values, dtype=float)

call_prices_pde = np.zeros_like(S_steps_values, dtype=float)
put_prices_pde = np.zeros_like(S_steps_values, dtype=float)

for j, s_steps in enumerate(S_steps_values):
    # Choose t_steps automatically so that the explicit scheme remains stable.
    t_steps_j = int(
        max(
            2000,
            np.ceil(3 * T * (sigma**2 * s_steps**2 + abs(r))),
        )
    )

    call_price_j, _, _, _ = bs_pde(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        call=1,
        S_max=S_max,
        S_steps=int(s_steps),
        t_steps=t_steps_j,
    )

    put_price_j, _, _, _ = bs_pde(
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        call=0,
        S_max=S_max,
        S_steps=int(s_steps),
        t_steps=t_steps_j,
    )

    call_prices_pde[j] = call_price_j
    put_prices_pde[j] = put_price_j

    call_errors[j] = abs(call_price_j - call_bs)
    put_errors[j] = abs(put_price_j - put_bs)

    print(
        f"S_steps={s_steps:4d}, "
        f"t_steps={t_steps_j:6d}, "
        f"call error={call_errors[j]:.8f}, "
        f"put error={put_errors[j]:.8f}"
    )

# The finest grid should be more accurate than the coarsest grid.
assert call_errors[-1] < call_errors[0]
assert put_errors[-1] < put_errors[0]


# --------------------------------------------------
# 7. Save convergence plot
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(S_steps_values, call_errors, marker="o", label="Call PDE error")
plt.plot(S_steps_values, put_errors, marker="o", label="Put PDE error")

plt.xlabel("Number of stock grid steps")
plt.ylabel("Absolute error at S0")
plt.title("Black-Scholes PDE Convergence")
plt.grid(alpha=0.3)
plt.legend()

save_path = os.path.join(plot_dir, "bs_pde_convergence.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Convergence plot saved to: {save_path}")

print("Black-Scholes PDE test passed.")