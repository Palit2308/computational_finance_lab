# Computational Finance Lab

`comp_fin_lab` is a modular Python package for computational finance, derivative pricing, stochastic-process simulation, and model diagnostics.

The repository refactors verified course and research implementations into reusable modules under `src/comp_fin_lab/`. The focus is on readable quantitative finance code, reproducible tests, and diagnostic plots that make numerical behaviour visible.

## Current Scope

The package currently includes:

- Vanilla option payoffs
- Black-Scholes European call and put pricing
- Black-Scholes path simulation
- Black-Scholes explicit finite-difference PDE solver
- Cox-Ross-Rubinstein European binomial pricing
- CRR lattice construction for diagnostics
- Down-and-out barrier call pricing
- Pricing by integration under Black-Scholes
- Closed-form, CRR, integration, and Laplace-based deltas
- American option valuation using CRR
- Perpetual American put valuation using an ODE/free-boundary approach
- Heston stochastic-volatility path simulation
- Black-Scholes and Heston characteristic functions
- Vanilla European call/put pricing by Laplace inversion
- Vanilla European call/put pricing by Fourier/FFT inversion
- Test scripts and diagnostic plots for validation

---

## 1. Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/Palit2308/computational_finance_lab.git
```

Basic import check:

```python
import comp_fin_lab

print("Package imported successfully")
```

Import commonly used modules:

```python
from comp_fin_lab.payoffs import vcall, vput, vopt, pcall
from comp_fin_lab.bs import eu_bs, bs_paths, bs_pde
from comp_fin_lab.binom import eu_crr, eu_crr_lattice
from comp_fin_lab.barriers import down_and_out_crr
from comp_fin_lab.integration import eu_int_sn
from comp_fin_lab.greeks import (
    eu_bs_delta,
    eu_crr_delta,
    delta_int_sn,
    delta_eu_lap,
    delta_eu_lap_bs,
    delta_eu_lap_heston,
)
from comp_fin_lab.heston import heston_paths
from comp_fin_lab.char_fun import bs_char, heston_char
from comp_fin_lab.laplace import eu_lap
from comp_fin_lab.fourier import eu_fourier
from comp_fin_lab.american import perp_am_put, am_crr
```

---

## 2. Project Structure

```text
computational_finance_lab/
│
├── src/
│   └── comp_fin_lab/
│       ├── __init__.py
│       ├── american.py
│       ├── barriers.py
│       ├── binom.py
│       ├── bs.py
│       ├── calibration.py
│       ├── char_fun.py
│       ├── fourier.py
│       ├── greeks.py
│       ├── heston.py
│       ├── integration.py
│       ├── laplace.py
│       ├── monte_carlo.py
│       ├── payoffs.py
│       └── rng.py
│
├── tests/
│   ├── plots/
│   ├── test_american.py
│   ├── test_barriers.py
│   ├── test_binom.py
│   ├── test_bs.py
│   ├── test_bs_paths.py
│   ├── test_bs_pde.py
│   ├── test_delta.py
│   ├── test_fourier.py
│   ├── test_heston_paths.py
│   ├── test_integration.py
│   ├── test_laplace.py
│   ├── test_packaging_works.py
│   └── test_payoffs.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

Source code lives in:

```text
src/comp_fin_lab/
```

Tests and reproducible diagnostic scripts live in:

```text
tests/
```

Generated plots are saved in:

```text
tests/plots/
```

---

## 3. Module Overview

### `payoffs.py`

Basic payoff functions.

```python
from comp_fin_lab.payoffs import vcall, vput, vopt, pcall

vcall(S=120, K=100)
vput(S=80, K=100)
vopt(S=120, K=100, call=True)
pcall(S=120, K=100, alpha=1.2)
```

Available functions:

```python
vcall(S, K)
vput(S, K)
vopt(S, K, call=True)
pcall(S, K, alpha)
```

---

### `bs.py`

Black-Scholes pricing, simulation, and PDE approximation.

Available functions:

```python
eu_bs(t, St, K, T, r, sigma, call)
bs_paths(S0, r, sigma, T, M, I, seed=None)
bs_pde(S0, K, T, r, sigma, call=1, S_max=None, S_steps=300, t_steps=20000)
```

European call and put pricing:

```python
from comp_fin_lab.bs import eu_bs

call_price = eu_bs(
    t=0,
    St=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=1,
)

put_price = eu_bs(
    t=0,
    St=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=0,
)
```

Black-Scholes path simulation:

```python
from comp_fin_lab.bs import bs_paths

S, S_hat, time_grid = bs_paths(
    S0=100,
    r=0.05,
    sigma=0.3,
    T=1,
    M=500,
    I=10000,
    seed=42,
)
```

`S_hat` is the discounted stock process. Under the risk-neutral measure, it should be approximately a martingale:

```text
E[S_hat(T)] = S0
```

Black-Scholes PDE approximation:

```python
from comp_fin_lab.bs import bs_pde

price, V, S_grid, tau_grid = bs_pde(
    S0=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=1,
    S_max=None,
    S_steps=300,
    t_steps=20000,
)
```

The PDE solver uses an explicit finite-difference scheme in time-to-maturity. If `S_max=None`, the upper stock boundary is set to:

```python
S_max = max(4 * K, 4 * S0)
```

---

### `binom.py`

Cox-Ross-Rubinstein European option pricing.

Available functions:

```python
eu_crr(g, S0, K, T, r, sigma, M, c=1)
eu_crr_lattice(g, S0, K, T, r, sigma, M, c=1)
```

Example:

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.binom import eu_crr, eu_crr_lattice

price = eu_crr(
    g=vcall,
    S0=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    M=100,
    c=1,
)

S, V, delta_t = eu_crr_lattice(
    g=vcall,
    S0=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    M=50,
    c=1,
)
```

`eu_crr` returns only the time-zero price. `eu_crr_lattice` returns the full stock-price lattice, option-value lattice, and time step.

---

### `barriers.py`

Down-and-out barrier call pricing using the CRR binomial model.

Available function:

```python
down_and_out_crr(g, S0, K, T, B, r, sigma, M, c=1)
```

Example:

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.barriers import down_and_out_crr

price = down_and_out_crr(
    g=vcall,
    S0=120,
    K=100,
    T=1,
    B=90,
    r=0.02,
    sigma=0.3,
    M=500,
    c=1,
)
```

---

### `integration.py`

European option pricing by numerical integration under the Black-Scholes model.

Available function:

```python
eu_int_sn(t, f, St, T, r, sigma, a, b)
```

Example:

```python
import numpy as np

from comp_fin_lab.payoffs import vcall
from comp_fin_lab.integration import eu_int_sn

price = eu_int_sn(
    t=0,
    f=lambda S: vcall(S, K=100),
    St=120,
    T=1,
    r=0.02,
    sigma=0.3,
    a=-np.inf,
    b=np.inf,
)
```

The payoff should be passed as a function of terminal stock price only. Use `lambda` if the payoff needs a strike or other parameters.

---

### `char_fun.py`

Characteristic functions used by Laplace and Fourier transform pricing.

Available functions:

```python
bs_char(u, St, t, T, r, sigma)
heston_char(u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho)
```

Example:

```python
from comp_fin_lab.char_fun import bs_char, heston_char
```

The pricing routines expect characteristic functions of `log(S_T)` with signature:

```python
char_fun(u, St, t, T, r, *input_params)
```

Black-Scholes example:

```python
phi_bs = bs_char(
    u=1.5,
    St=120,
    t=0,
    T=1,
    r=0.02,
    sigma=0.3,
)
```

Heston example:

```python
phi_heston = heston_char(
    u=1.5,
    St=120,
    t=0,
    T=1,
    r=0.02,
    gamt=0.3**2,
    kappa=0.3**2,
    lamb=2.5,
    sig_tilde=0.3,
    rho=-0.5,
)
```

In the Heston characteristic function, `gamt` is the current variance. The parameterization used is:

```text
dgamma_t = (kappa - lamb * gamma_t) dt
           + sig_tilde * sqrt(gamma_t) dW_t
```

so the long-run variance is:

```text
kappa / lamb
```

---

### `laplace.py`

Vanilla European call/put pricing by Laplace inversion.

Available function:

```python
eu_lap(char_fun, K_arr, R, St, t, T, r, *input_params)
```

Use:

```text
R > 1   for calls
R < 0   for puts
```

The function is generic across models as long as the model supplies a characteristic function of `log(S_T)`.

Black-Scholes example:

```python
import numpy as np

from comp_fin_lab.char_fun import bs_char
from comp_fin_lab.laplace import eu_lap

K_arr = np.linspace(50, 300, 100)

call_BS_Laplace = eu_lap(
    bs_char,
    K_arr,
    2,
    120,
    0,
    1,
    0.02,
    0.3,
)
```

Heston example:

```python
from comp_fin_lab.char_fun import heston_char
from comp_fin_lab.laplace import eu_lap

call_Heston_Laplace = eu_lap(
    heston_char,
    K_arr,
    2,
    120,
    0,
    1,
    0.02,
    0.3**2,
    0.3**2,
    2.5,
    0.3,
    -0.5,
)
```

Put example:

```python
put_Heston_Laplace = eu_lap(
    heston_char,
    K_arr,
    -1,
    120,
    0,
    1,
    0.02,
    0.3**2,
    0.3**2,
    2.5,
    0.3,
    -0.5,
)
```

---

### `fourier.py`

Vanilla European call/put pricing by Fourier inversion using FFT.

Available function:

```python
eu_fourier(char_fun, K_arr, R, St, t, T, r, M, N, *input_params)
```

Use:

```text
R > 1   for calls
R < 0   for puts
```

`M` controls the Fourier-domain truncation and `N` is the number of FFT grid points.

Black-Scholes example:

```python
import numpy as np

from comp_fin_lab.char_fun import bs_char
from comp_fin_lab.fourier import eu_fourier

K_arr = np.linspace(50, 300, 100)

call_BS_Fourier = eu_fourier(
    bs_char,
    K_arr,
    2,
    120,
    0,
    1,
    0.02,
    5000,
    2**15,
    0.3,
)
```

Heston example:

```python
from comp_fin_lab.char_fun import heston_char
from comp_fin_lab.fourier import eu_fourier

call_Heston_Fourier = eu_fourier(
    heston_char,
    K_arr,
    2,
    120,
    0,
    1,
    0.02,
    5000,
    2**15,
    0.3**2,
    0.3**2,
    2.5,
    0.3,
    -0.5,
)
```

FFT pricing is designed for pricing many strikes at once. For a strike vector, it should be faster than evaluating one numerical quadrature per strike.

---

### `greeks.py`

Delta calculations using closed-form, integration, CRR, and Laplace-transform methods.

Available functions:

```python
eu_crr_delta(g, S0, K, T, r, sigma, M, c=1)
eu_bs_delta(t, St, K, T, r, sigma, call)
delta_int_sn(t, f_prime, St, T, r, sigma, a, b)

delta_eu_lap_bs(u, St, t, T, r, sig_tilde)
delta_eu_lap_heston(u, St, t, T, r, gamt, kappa, lamb, sig_tilde, rho)
delta_eu_lap(delta_char_fun, K_arr, R, St, t, T, r, *input_params)
```

Closed-form Black-Scholes delta:

```python
from comp_fin_lab.greeks import eu_bs_delta

call_delta = eu_bs_delta(
    t=0,
    St=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=1,
)

put_delta = eu_bs_delta(
    t=0,
    St=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=0,
)
```

CRR delta:

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.greeks import eu_crr_delta

delta = eu_crr_delta(
    g=vcall,
    S0=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    M=1000,
)
```

Delta by integration:

```python
import numpy as np

from comp_fin_lab.greeks import delta_int_sn

def vcall_prime(S, K):
    return 1.0 if S > K else 0.0

delta = delta_int_sn(
    t=0,
    f_prime=lambda S: vcall_prime(S, 100),
    St=120,
    T=1,
    r=0.02,
    sigma=0.3,
    a=-np.inf,
    b=np.inf,
)
```

Delta by Laplace inversion:

```python
from comp_fin_lab.greeks import delta_eu_lap, delta_eu_lap_bs, delta_eu_lap_heston

K_arr = np.linspace(50, 300, 100)

bs_call_delta = delta_eu_lap(
    delta_eu_lap_bs,
    K_arr,
    2,
    120,
    0,
    1,
    0.02,
    0.3,
)

heston_call_delta = delta_eu_lap(
    delta_eu_lap_heston,
    K_arr,
    2,
    120,
    0,
    1,
    0.02,
    0.3**2,
    0.3**2,
    2.5,
    0.3,
    -0.5,
)
```

---

### `american.py`

American option valuation routines.

Available functions:

```python
perp_am_put(K, r, sigma, S_min=1e-6, S_max=200, n_grid=200, S_grid=None)
am_crr(S_ini, K, T, r, sigma, N, opttype, c=1)
```

Perpetual American put:

```python
from comp_fin_lab.american import perp_am_put

S_grid, v_grid, x_star = perp_am_put(
    K=100,
    r=0.02,
    sigma=0.3,
    S_min=1e-6,
    S_max=200,
    n_grid=200,
)
```

Finite-maturity American option using CRR:

```python
from comp_fin_lab.american import am_crr

price, C, S = am_crr(
    S_ini=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    N=500,
    opttype="P",
    c=1,
)
```

Use:

```text
opttype = "C"  -> American call
opttype = "P"  -> American put
```

---

### `heston.py`

Heston stochastic-volatility path simulation.

Available functions:

```python
random_number_gen(M, I, seed=None)
heston_paths(S0, r, gamma0, kappa, theta, sigma, rho, T, M, I, seed=None, rand=None)
```

Internal helpers:

```python
_cholesky_from_rho(rho)
_sde_gamma(gamma0, kappa, theta, sigma, T, M, I, rand, rho)
```

Example:

```python
from comp_fin_lab.heston import heston_paths

S, gamma = heston_paths(
    S0=100,
    r=0.05,
    gamma0=0.04,
    kappa=2,
    theta=0.04,
    sigma=0.3,
    rho=-0.9,
    T=1,
    M=500,
    I=10000,
    seed=42,
)

volatility = np.sqrt(gamma)
```

The simulated model is:

```text
dS(t) = r S(t) dt + sqrt(gamma(t)) S(t) dW_S(t)

dgamma(t) = kappa(theta - gamma(t)) dt
            + sigma sqrt(gamma(t)) dW_gamma(t)

corr(dW_S(t), dW_gamma(t)) = rho
```

The implementation uses log-Euler updates for stock prices, Euler-type updates for variance, non-negativity truncation for variance, and Cholesky correlation for the Brownian shocks.

---

## 4. Running Tests

The tests are executable Python scripts. From the project root:

```bash
cd tests

python test_packaging_works.py
python test_payoffs.py
python test_binom.py
python test_bs.py
python test_bs_paths.py
python test_bs_pde.py
python test_barriers.py
python test_integration.py
python test_delta.py
python test_american.py
python test_heston_paths.py
python test_laplace.py
python test_fourier.py
```

The tests check numerical sanity, no-arbitrage restrictions, convergence behaviour, parity relationships, reproducibility, and generated plots.

---

## 5. Test Coverage

### Packaging and payoffs

```text
test_packaging_works.py
test_payoffs.py
```

Checks that the package imports correctly and that payoff functions work for scalar and vectorized inputs.

### Binomial and Black-Scholes pricing

```text
test_binom.py
test_bs.py
```

Checks CRR pricing, full lattice construction, volatility sensitivity, put-call parity, and CRR approximation errors against Black-Scholes.

Generated plots:

```text
eu_crr_stock_and_call_structures_with_one_random_path.png
eu_crr_sigma_sensitivity_parity.png
bs_crr_anchored_error.png
```

### Black-Scholes simulation and PDE

```text
test_bs_paths.py
test_bs_pde.py
```

Checks Black-Scholes path shapes, positivity, reproducibility, martingale behaviour of discounted stock prices, terminal log-return moments, explicit PDE prices, boundary conditions, and convergence.

Generated plots:

```text
bs_stock_and_discounted_paths.png
bs_terminal_log_return_distribution.png
bs_pde_vs_black_scholes_prices.png
bs_pde_absolute_pricing_error.png
bs_pde_convergence.png
```

### Barrier, integration, and American options

```text
test_barriers.py
test_integration.py
test_american.py
```

Checks down-and-out barrier behaviour, integration prices against Black-Scholes, power-call pricing against CRR, runtime comparison, perpetual American put diagnostics, finite-maturity American CRR pricing, and early-exercise behaviour.

Generated plots:

```text
down_and_out_crr_barrier_sensitivity.png
integration_vs_crr_power_call_runtime.png
perpetual_american_put.png
american_crr_convergence.png
american_crr_early_exercise_premium.png
american_crr_sigma_sensitivity.png
```

### Delta calculations

```text
test_delta.py
```

Checks:

- CRR delta against Black-Scholes delta
- Integration delta against Black-Scholes delta
- Put-call delta parity
- Black-Scholes Laplace delta against closed-form delta
- Heston Laplace call/put delta sanity checks
- Heston Laplace delta parity

Generated plots:

```text
bs_laplace_call_delta.png
bs_laplace_put_delta.png
heston_laplace_call_delta.png
heston_laplace_put_delta.png
```

### Heston path simulation

```text
test_heston_paths.py
```

Checks stock-path and variance-path shapes, initial conditions, positivity/non-negativity, finite values, reproducibility, and volatility-path diagnostics.

Generated plot:

```text
heston_price_and_volatility_paths.png
```

### Laplace pricing

```text
test_laplace.py
```

Checks Black-Scholes and Heston vanilla call/put prices from Laplace inversion.

The tests verify:

- finite prices
- non-negative prices
- no-arbitrage upper/lower bounds
- monotonicity in strike
- convexity in strike
- put-call parity

Generated plots:

```text
heston_laplace_call_prices.png
heston_laplace_put_prices.png
bs_laplace_call_prices.png
bs_laplace_put_prices.png
bs_laplace_put_call_parity_error.png
```

### Fourier/FFT pricing

```text
test_fourier.py
```

Checks Black-Scholes and Heston vanilla call/put prices from Fourier/FFT inversion.

The tests verify:

- finite prices
- no-arbitrage bounds
- monotonicity in strike
- convexity in strike
- put-call parity
- Fourier/FFT runtime compared with Laplace quadrature for the same strike vector

Generated plots:

```text
heston_fourier_call_prices.png
heston_fourier_put_prices.png
heston_fourier_put_call_parity_error.png
bs_fourier_call_prices.png
bs_fourier_put_prices.png
bs_fourier_put_call_parity_error.png
```

---

## 6. Example Workflow

A typical workflow is:

```python
import numpy as np

from comp_fin_lab.char_fun import heston_char
from comp_fin_lab.laplace import eu_lap
from comp_fin_lab.fourier import eu_fourier
from comp_fin_lab.greeks import delta_eu_lap, delta_eu_lap_heston

K_arr = np.linspace(50, 300, 100)

S0 = 120
t = 0
T = 1
r = 0.02

gam0 = 0.3**2
kappa = 0.3**2
lamb = 2.5
sig_tilde = 0.3
rho = -0.5

call_laplace = eu_lap(
    heston_char,
    K_arr,
    2,
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

call_fourier = eu_fourier(
    heston_char,
    K_arr,
    2,
    S0,
    t,
    T,
    r,
    5000,
    2**15,
    gam0,
    kappa,
    lamb,
    sig_tilde,
    rho,
)

call_delta = delta_eu_lap(
    delta_eu_lap_heston,
    K_arr,
    2,
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
```

---

## 7. Mathematical Scope

The repository currently covers:

- Risk-neutral valuation
- Black-Scholes closed-form pricing
- Black-Scholes path simulation
- Black-Scholes finite-difference PDE approximation
- CRR binomial approximation
- Pricing by risk-neutral integration
- Characteristic-function based pricing
- Laplace inversion for vanilla European options
- Fourier/FFT inversion for vanilla European options
- Delta calculations from multiple methods
- Heston stochastic volatility simulation
- American option valuation by dynamic programming
- Perpetual American put valuation by ODE/free-boundary methods

The characteristic-function pricing routines are generic across models, but currently hard-code the vanilla European call/put payoff transform.

---

## 8. Generated Plots

Generated plots are saved under:

```text
tests/plots/
```

---

## 9. Future Extensions

Planned extensions include:

- Monte Carlo option pricing
- Random number generation methods
- Inverse transform sampling
- Acceptance-rejection sampling
- Ornstein-Uhlenbeck process simulation
- CIR process simulation
- Additional affine models
- Heston calibration
- Implied volatility routines
- Calibration diagnostics
- Delta hedging backtests
- Fourier pricing refinements
- Documentation pages and example notebooks
