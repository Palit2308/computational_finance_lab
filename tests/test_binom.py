import os

import numpy as np
import matplotlib.pyplot as plt

from comp_fin_lab.payoffs import vcall, vput
from comp_fin_lab.binom import eu_crr, eu_crr_lattice


# --------------------------------------------------
# Parameters
# --------------------------------------------------

r = 0.02
sigma = 0.3
T = 1
K = 100
S0 = 120
M = 50
c = 1
sigmas = np.linspace(0.01, 5, 500)

# --------------------------------------------------
# Compute CRR price and full lattice
# --------------------------------------------------

price = eu_crr(
    g=vcall,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
    c=c,
)

S, V, delta_t = eu_crr_lattice(
    g=vcall,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
    c=c,
)

print(f"European call price from eu_crr: {price:.6f}")
print(f"European call price from lattice V[0, 0]: {V[0, 0]:.6f}")


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Draw one random binomial path
# --------------------------------------------------

rng = np.random.default_rng(24)

coin_flips = rng.binomial(1, 0.5, M)
binomial_path = coin_flips.cumsum()
binomial_path = np.append(0, binomial_path)


# --------------------------------------------------
# Plot stock price lattice and option value lattice
# --------------------------------------------------

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

for i in range(M + 1):
    ax[0].plot(
        np.repeat(i * delta_t, i + 1),
        S[: i + 1, i],
        "+",
        color="black",
        markersize=4,
    )

    ax[1].plot(
        np.repeat(i * delta_t, i + 1),
        V[: i + 1, i],
        "+",
        color="black",
        markersize=4,
    )

# Plot one random path through the stock lattice
ax[0].plot(
    np.linspace(0, T, M + 1),
    S[binomial_path, np.arange(M + 1)],
    color="red",
)

# Plot the corresponding option value path
ax[1].plot(
    np.linspace(0, T, M + 1),
    V[binomial_path, np.arange(M + 1)],
    color="red",
)

ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)

ax[0].set_xlabel("$t$")
ax[1].set_xlabel("$t$")

ax[0].set_ylabel("$S(t)$")
ax[1].set_ylabel("$V(t)$")

ax[0].set_title("CRR Stock Price")
ax[1].set_title("CRR Call Option Value")

fig.suptitle("European Call Option under the CRR Binomial Model")

save_path = os.path.join(plot_dir, "eu_crr_stock_and_call_structures_with_one_random_path.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"Plot saved to: {save_path}")


# --------------------------------------------------
# Compute CRR prices for different volatilities
# --------------------------------------------------

call_prices_CRR = np.zeros_like(sigmas)
put_prices_CRR = np.zeros_like(sigmas)

for i in range(len(sigmas)):
    call_prices_CRR[i] = eu_crr(
        g=vcall,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigmas[i],
        M=M,
        c=c,
    )

    put_prices_CRR[i] = eu_crr(
        g=vput,
        S0=S0,
        K=K,
        T=T,
        r=r,
        sigma=sigmas[i],
        M=M,
        c=c,
    )


# --------------------------------------------------
# No-arbitrage bounds and put-call parity
# --------------------------------------------------

discounted_strike = K * np.exp(-r * T)

call_lower_bound = S0 - discounted_strike
call_upper_bound = S0

put_lower_bound = 0
put_upper_bound = discounted_strike

put_call_parity_error = put_prices_CRR - call_prices_CRR - discounted_strike + S0


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Plot diagnostics
# --------------------------------------------------

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Call prices
ax[0].plot(sigmas, call_prices_CRR)
ax[0].axhline(call_lower_bound, linestyle="--", label=r"$S_0 - Ke^{-rT}$")
ax[0].axhline(call_upper_bound, linestyle="--", label=r"$S_0$")
ax[0].set_xlabel(r"$\sigma$")
ax[0].set_ylabel("Call price")
ax[0].set_title("CRR Call Price vs Volatility")
ax[0].grid(alpha=0.3)
ax[0].legend()

# Put prices
ax[1].plot(sigmas, put_prices_CRR)
ax[1].axhline(put_upper_bound, linestyle="--", label=r"$Ke^{-rT}$")
ax[1].axhline(put_lower_bound, linestyle="--", label="0")
ax[1].set_xlabel(r"$\sigma$")
ax[1].set_ylabel("Put price")
ax[1].set_title("CRR Put Price vs Volatility")
ax[1].grid(alpha=0.3)
ax[1].legend()

# Put-call parity error
ax[2].plot(sigmas, put_call_parity_error)
ax[2].axhline(0, linestyle="--")
ax[2].set_xlabel(r"$\sigma$")
ax[2].set_ylabel(r"$P - C - Ke^{-rT} + S_0$")
ax[2].set_title("Put-Call Parity Error")
ax[2].grid(alpha=0.3)

fig.suptitle("CRR Volatility Sensitivity and Put-Call Parity Check", fontsize=14)
fig.tight_layout()

save_path = os.path.join(plot_dir, "eu_crr_sigma_sensitivity_parity.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to: {save_path}")
print(f"Maximum absolute put-call parity error: {np.max(np.abs(put_call_parity_error)):.10f}")