import numpy as np

from comp_fin_lab.payoffs import vcall, vput
from comp_fin_lab.greeks import eu_crr_delta, eu_bs_delta, delta_int_sn


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