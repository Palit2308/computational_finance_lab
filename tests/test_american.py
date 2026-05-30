import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.american import perp_am_put
from comp_fin_lab.bs import eu_bs


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