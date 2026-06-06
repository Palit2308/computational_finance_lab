import numpy as np
import os
import matplotlib.pyplot as plt
from comp_fin_lab.payoffs import vcall, vput
from comp_fin_lab.greeks import eu_crr_delta, eu_bs_delta, delta_int_sn, delta_eu_lap, delta_eu_lap_heston, delta_eu_lap_bs


# --------------------------------------------------
# Payoff derivatives
# --------------------------------------------------

def vcall_prime(S, K):
    """
    Derivative of call payoff max(S - K, 0) with respect to S.
    """
    return 1.0 if S > K else 0.0


def vput_prime(S, K):
    """
    Derivative of put payoff max(K - S, 0) with respect to S.
    """
    return -1.0 if S < K else 0.0


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
c = 1

a = -np.inf
b = np.inf


# --------------------------------------------------
# Call delta: CRR vs BS vs integration
# --------------------------------------------------

call_delta_crr = eu_crr_delta(
    g=vcall,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
    c=c,
)

call_delta_bs = eu_bs_delta(
    t=t,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
)

call_delta_int = delta_int_sn(
    t=t,
    f_prime=lambda S: vcall_prime(S, K),
    St=S0,
    T=T,
    r=r,
    sigma=sigma,
    a=a,
    b=b,
)


print("Call delta CRR:        ", call_delta_crr)
print("Call delta BS:         ", call_delta_bs)
print("Call delta integration:", call_delta_int)

print("Call |CRR - BS|:       ", abs(call_delta_crr - call_delta_bs))
print("Call |INT - BS|:       ", abs(call_delta_int - call_delta_bs))


assert np.isfinite(call_delta_crr)
assert np.isfinite(call_delta_bs)
assert np.isfinite(call_delta_int)

assert 0 <= call_delta_bs <= 1
assert 0 <= call_delta_crr <= 1
assert 0 <= call_delta_int <= 1

assert abs(call_delta_int - call_delta_bs) < 1e-4
assert abs(call_delta_crr - call_delta_bs) < 5e-3


# --------------------------------------------------
# Put delta: CRR vs BS vs integration
# --------------------------------------------------

put_delta_crr = eu_crr_delta(
    g=vput,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
    c=c,
)

put_delta_bs = eu_bs_delta(
    t=t,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=0,
)

put_delta_int = delta_int_sn(
    t=t,
    f_prime=lambda S: vput_prime(S, K),
    St=S0,
    T=T,
    r=r,
    sigma=sigma,
    a=a,
    b=b,
)


print("\nPut delta CRR:        ", put_delta_crr)
print("Put delta BS:         ", put_delta_bs)
print("Put delta integration:", put_delta_int)

print("Put |CRR - BS|:       ", abs(put_delta_crr - put_delta_bs))
print("Put |INT - BS|:       ", abs(put_delta_int - put_delta_bs))


assert np.isfinite(put_delta_crr)
assert np.isfinite(put_delta_bs)
assert np.isfinite(put_delta_int)

assert -1 <= put_delta_bs <= 0
assert -1 <= put_delta_crr <= 0
assert -1 <= put_delta_int <= 0

assert abs(put_delta_int - put_delta_bs) < 1e-4
assert abs(put_delta_crr - put_delta_bs) < 5e-3


# --------------------------------------------------
# Put-call delta parity
# --------------------------------------------------
# From put-call parity:
# C - P = S - K exp(-rT)
# Differentiating with respect to S:
# Delta_call - Delta_put = 1

print("\nDelta parity checks")
print("BS call delta - put delta: ", call_delta_bs - put_delta_bs)
print("INT call delta - put delta:", call_delta_int - put_delta_int)
print("CRR call delta - put delta:", call_delta_crr - put_delta_crr)

assert abs((call_delta_bs - put_delta_bs) - 1) < 1e-10
assert abs((call_delta_int - put_delta_int) - 1) < 1e-4
assert abs((call_delta_crr - put_delta_crr) - 1) < 5e-3


print("\nDelta test passed.")

# --------------------------------------------------
# Laplace delta tests and plots: Black-Scholes and Heston
# --------------------------------------------------

K_arr = np.linspace(50, 300, 100)

R_call = 2
R_put = -1

gam0 = 0.3**2
kappa_heston = 0.3**2
lamb_heston = 2.5
sig_tilde_heston = 0.3
rho_heston = -0.5


# --------------------------------------------------
# Black-Scholes deltas by Laplace inversion
# --------------------------------------------------

call_delta_bs_lap = delta_eu_lap(
    delta_eu_lap_bs,
    K_arr,
    R_call,
    S0,
    t,
    T,
    r,
    sigma,
)

put_delta_bs_lap = delta_eu_lap(
    delta_eu_lap_bs,
    K_arr,
    R_put,
    S0,
    t,
    T,
    r,
    sigma,
)


# --------------------------------------------------
# Black-Scholes closed-form benchmark deltas
# --------------------------------------------------

call_delta_bs_closed = np.array(
    [
        eu_bs_delta(
            t=t,
            St=S0,
            K=K_i,
            T=T,
            r=r,
            sigma=sigma,
            call=1,
        )
        for K_i in K_arr
    ]
)

put_delta_bs_closed = np.array(
    [
        eu_bs_delta(
            t=t,
            St=S0,
            K=K_i,
            T=T,
            r=r,
            sigma=sigma,
            call=0,
        )
        for K_i in K_arr
    ]
)


# --------------------------------------------------
# Black-Scholes Laplace delta checks
# --------------------------------------------------

assert call_delta_bs_lap.shape == K_arr.shape
assert put_delta_bs_lap.shape == K_arr.shape

assert np.all(np.isfinite(call_delta_bs_lap))
assert np.all(np.isfinite(put_delta_bs_lap))

assert np.all(call_delta_bs_lap >= -1e-8)
assert np.all(call_delta_bs_lap <= 1 + 1e-8)

assert np.all(put_delta_bs_lap >= -1 - 1e-8)
assert np.all(put_delta_bs_lap <= 1e-8)

assert np.all(np.diff(call_delta_bs_lap) <= 1e-6)
assert np.all(np.diff(put_delta_bs_lap) <= 1e-6)

bs_call_delta_error = np.max(np.abs(call_delta_bs_lap - call_delta_bs_closed))
bs_put_delta_error = np.max(np.abs(put_delta_bs_lap - put_delta_bs_closed))

assert bs_call_delta_error < 1e-4
assert bs_put_delta_error < 1e-4

bs_delta_parity_error = np.max(
    np.abs((call_delta_bs_lap - put_delta_bs_lap) - 1)
)

assert bs_delta_parity_error < 1e-4


# --------------------------------------------------
# Heston deltas by Laplace inversion
# --------------------------------------------------

call_delta_heston_lap = delta_eu_lap(
    delta_eu_lap_heston,
    K_arr,
    R_call,
    S0,
    t,
    T,
    r,
    gam0,
    kappa_heston,
    lamb_heston,
    sig_tilde_heston,
    rho_heston,
)

put_delta_heston_lap = delta_eu_lap(
    delta_eu_lap_heston,
    K_arr,
    R_put,
    S0,
    t,
    T,
    r,
    gam0,
    kappa_heston,
    lamb_heston,
    sig_tilde_heston,
    rho_heston,
)


# --------------------------------------------------
# Heston Laplace delta checks
# --------------------------------------------------

assert call_delta_heston_lap.shape == K_arr.shape
assert put_delta_heston_lap.shape == K_arr.shape

assert np.all(np.isfinite(call_delta_heston_lap))
assert np.all(np.isfinite(put_delta_heston_lap))

assert np.all(call_delta_heston_lap >= -1e-6)
assert np.all(call_delta_heston_lap <= 1 + 1e-6)

assert np.all(put_delta_heston_lap >= -1 - 1e-6)
assert np.all(put_delta_heston_lap <= 1e-6)

assert np.all(np.diff(call_delta_heston_lap) <= 1e-5)
assert np.all(np.diff(put_delta_heston_lap) <= 1e-5)

heston_delta_parity_error = np.max(
    np.abs((call_delta_heston_lap - put_delta_heston_lap) - 1)
)

assert heston_delta_parity_error < 1e-4


# --------------------------------------------------
# Create plots folder in current directory
# --------------------------------------------------

plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)


# --------------------------------------------------
# Plot Black-Scholes call delta
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, call_delta_bs_lap, label="BS call delta by Laplace")
plt.plot(K_arr, call_delta_bs_closed, linestyle="--", label="BS closed-form call delta")

plt.xlabel("Strike values")
plt.ylabel("Delta")
plt.title("Black-Scholes Call Delta vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_call_delta = os.path.join(plot_dir, "bs_laplace_call_delta.png")
plt.savefig(save_path_bs_call_delta, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Black-Scholes put delta
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, put_delta_bs_lap, label="BS put delta by Laplace")
plt.plot(K_arr, put_delta_bs_closed, linestyle="--", label="BS closed-form put delta")

plt.xlabel("Strike values")
plt.ylabel("Delta")
plt.title("Black-Scholes Put Delta vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_bs_put_delta = os.path.join(plot_dir, "bs_laplace_put_delta.png")
plt.savefig(save_path_bs_put_delta, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Heston call delta
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, call_delta_heston_lap, label="Heston call delta by Laplace")

plt.xlabel("Strike values")
plt.ylabel("Delta")
plt.title("Heston Call Delta vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_heston_call_delta = os.path.join(plot_dir, "heston_laplace_call_delta.png")
plt.savefig(save_path_heston_call_delta, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Plot Heston put delta
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_arr, put_delta_heston_lap, label="Heston put delta by Laplace")

plt.xlabel("Strike values")
plt.ylabel("Delta")
plt.title("Heston Put Delta vs Strike using Laplace Inversion")
plt.grid(alpha=0.3)
plt.legend()

save_path_heston_put_delta = os.path.join(plot_dir, "heston_laplace_put_delta.png")
plt.savefig(save_path_heston_put_delta, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

print("\nLaplace delta checks")

print("Black-Scholes Laplace call delta min:", call_delta_bs_lap.min())
print("Black-Scholes Laplace call delta max:", call_delta_bs_lap.max())

print("Black-Scholes Laplace put delta min: ", put_delta_bs_lap.min())
print("Black-Scholes Laplace put delta max: ", put_delta_bs_lap.max())

print("BS max call delta error vs closed form:", bs_call_delta_error)
print("BS max put delta error vs closed form: ", bs_put_delta_error)
print("BS Laplace delta parity error:         ", bs_delta_parity_error)

print("\nHeston Laplace call delta min:", call_delta_heston_lap.min())
print("Heston Laplace call delta max:", call_delta_heston_lap.max())

print("Heston Laplace put delta min: ", put_delta_heston_lap.min())
print("Heston Laplace put delta max: ", put_delta_heston_lap.max())

print("Heston Laplace delta parity error:", heston_delta_parity_error)

print(f"\nBS call delta plot saved to: {save_path_bs_call_delta}")
print(f"BS put delta plot saved to: {save_path_bs_put_delta}")
print(f"Heston call delta plot saved to: {save_path_heston_call_delta}")
print(f"Heston put delta plot saved to: {save_path_heston_put_delta}")

print("\nLaplace delta plots saved.")