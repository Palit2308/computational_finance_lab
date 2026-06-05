# Computational Finance Lab

`comp_fin_lab` is a modular Python package for computational finance, derivative pricing, and stochastic modelling.

The project started from verified computational finance implementations and refactors them into a reusable Python package. The goal is to preserve mathematically checked course implementations while making them usable as importable modules in future research, modelling, and portfolio projects.

The package currently includes:

- Vanilla option payoffs
- Black-Scholes European call and put pricing
- Cox-Ross-Rubinstein binomial tree pricing
- CRR lattice construction for diagnostics and visualisation
- Down-and-out barrier call pricing
- Pricing by integration under the Black-Scholes model
- Delta calculations using Black-Scholes, integration, and CRR methods
- American option valuation in the CRR model
- Perpetual American put valuation using an ODE/free-boundary approach
- BS  path simulation
- BS PDE solver
- Heston path simulation
- Test scripts and diagnostic plots for model validation

---

## 1. Installation and Usage

### Install directly from GitHub

Once the repository is public, the package can be installed from GitHub using:

```bash
pip install git+https://github.com/Palit2308/computational_finance_lab.git

```

### Basic import test

After installation, check that the package imports correctly:

```python
import comp_fin_lab

print("Package imported successfully")
```

You can also import individual modules:

```python
from comp_fin_lab.payoffs import vcall, vput, vopt, pcall
from comp_fin_lab.bs import eu_bs
from comp_fin_lab.binom import eu_crr, eu_crr_lattice
from comp_fin_lab.barriers import down_and_out_crr
from comp_fin_lab.integration import eu_int_sn
from comp_fin_lab.greeks import eu_bs_delta, eu_crr_delta, delta_int_sn
from comp_fin_lab.american import perp_am_put, am_crr
```

## 2. Project Structure

Current project structure:

```txt
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
│   │   ├── american_crr_convergence.png
│   │   ├── american_crr_early_exercise_premium.png
│   │   ├── american_crr_sigma_sensitivity.png
│   │   ├── bs_crr_anchored_error.png
│   │   ├── down_and_out_crr_barrier_sensitivity.png
│   │   ├── eu_crr_sigma_sensitivity_parity.png
│   │   ├── eu_crr_stock_and_call_structures_with_one_random_path.png
│   │   ├── integration_vs_crr_power_call_runtime.png
│   │   └── perpetual_american_put.png
│   │
│   ├── test_american.py
│   ├── test_barriers.py
│   ├── test_binom.py
│   ├── test_bs.py
│   ├── test_delta.py
│   ├── test_integration.py
│   ├── test_packaging_works.py
│   └── test_payoffs.py
│
├── pyproject.toml
├── README.md
└── .gitignore
```

The source code lives inside:

```txt
src/comp_fin_lab/
```

The tests and reproducible examples live inside:

```txt
tests/
```

The generated diagnostic plots are saved in:

```txt
tests/plots/
```

## 3. Module Overview

### `payoffs.py`

Contains basic option payoff functions.

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

Contains the Black-Scholes formula for European call and put options.

```python
from comp_fin_lab.bs import eu_bs

price_call = eu_bs(
    t=0,
    St=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=1,
)

price_put = eu_bs(
    t=0,
    St=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    call=0,
)
```

Available function:

```python
eu_bs(t, St, K, T, r, sigma, call)
```

where:

```text
call = 1  -> European call
call = 0  -> European put
```

Simulating stock price paths under the Black-Scholes model:

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

The simulation first constructs the discounted stock price process

```text
S_hat(t) = exp(-rt) S(t)
```

under the risk-neutral measure, and then undiscounts it internally to obtain the stock price process `S(t)`.

Under the risk-neutral Black-Scholes model:

```text
dS(t) = r S(t) dt + sigma S(t) dW(t)
```

The exact solution is:

```text
S(t) = S0 exp((r - 0.5 sigma^2)t + sigma W(t))
```

Parameters:

```text
S0 : float
    Initial stock price.

r : float
    Risk-free interest rate.

sigma : float
    Constant volatility.

T : float
    Time horizon.

M : int
    Number of time steps.

I : int
    Number of simulated paths.

seed : int, optional
    Random seed for reproducibility.
```

Returns:

```text
S : ndarray
    Undiscounted Black-Scholes stock price paths with shape (M + 1, I).

S_hat : ndarray
    Discounted stock price paths with shape (M + 1, I).

time_grid : ndarray
    Time grid with shape (M + 1,).
```

The discounted process is useful because, under the risk-neutral measure, it should be approximately a martingale:

```text
E[S_hat(T)] = S0
```

The undiscounted process should satisfy:

```text
E[S(T)] = S0 exp(rT)
```

Black-Scholes PDE approximation : 

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

The function prices a European call or put option by solving the Black-Scholes PDE using an explicit finite-difference scheme.

The Black-Scholes PDE is solved in time-to-maturity:

```text
tau = T - t
```

The PDE is:

```text
dV/dtau = 0.5 sigma^2 S^2 d2V/dS2
          + r S dV/dS
          - r V
```

with initial condition at `tau = 0`:

```text
V(0, S) = payoff(S)
```

For a call option:

```text
V(0, S) = max(S - K, 0)
```

For a put option:

```text
V(0, S) = max(K - S, 0)
```

Parameters:

```text
S0 : float
    Initial stock price where the option price is evaluated.

K : float
    Strike price.

T : float
    Time to maturity.

r : float
    Risk-free interest rate.

sigma : float
    Volatility.

call : int
    1 for call option, 0 for put option.

S_max : float, optional
    Upper boundary of stock-price grid.
    If None, set to max(4*K, 4*S0).

S_steps : int, optional
    Number of stock-price grid steps.

t_steps : int, optional
    Number of time steps in time-to-maturity.
```

Returns:

```text
price : float
    Interpolated option price at S0.

V : ndarray
    PDE value grid with shape (t_steps + 1, S_steps + 1).
    Rows correspond to time-to-maturity tau.
    Columns correspond to stock-price grid points.

S_grid : ndarray
    Stock-price grid.

tau_grid : ndarray
    Time-to-maturity grid.
```

If `S_max=None`, the upper stock-price boundary is chosen as:

```python
S_max = max(4 * K, 4 * S0)
```

Because the method uses an explicit finite-difference scheme, the function includes a practical stability check. If the time step is too large relative to the stock grid size, the function raises a `ValueError` and suggests increasing `t_steps` or reducing `S_steps`.


---

### `binom.py`

Contains the Cox-Ross-Rubinstein binomial tree implementation.

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.binom import eu_crr

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
```

Available functions:

```python
eu_crr(g, S0, K, T, r, sigma, M, c=1)
eu_crr_lattice(g, S0, K, T, r, sigma, M, c=1)
```

`eu_crr` returns only the time-zero option price.

`eu_crr_lattice` returns the full stock-price lattice, option-value lattice, and time step:

```python
S, V, delta_t = eu_crr_lattice(
    g=vcall,
    S0=120,
    K=100,
    T=1,
    r=0.02,
    sigma=0.3,
    M=50,
)
```

---

### `barriers.py`

Contains down-and-out barrier option pricing using the CRR binomial model.

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
)
```

Available function:

```python
down_and_out_crr(g, S0, K, T, B, r, sigma, M, c=1)
```

---

### `integration.py`

Contains pricing by integration under the Black-Scholes model.

The function evaluates the discounted risk-neutral expectation by integrating over a standard normal variable.

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.integration import eu_int_sn

price = eu_int_sn(
    t=0,
    f=lambda S: vcall(S, 100),
    St=120,
    T=1,
    r=0.02,
    sigma=0.3,
    a=-float("inf"),
    b=float("inf"),
)
```

Available function:

```python
eu_int_sn(t, f, St, T, r, sigma, a, b)
```

Here `f` should be a function of terminal stock price only. If the payoff requires a strike, wrap it using `lambda`.

Example:

```python
f = lambda S: vcall(S, K=100)
```

---

### `greeks.py`

Contains delta calculations using three methods:

1. Black-Scholes closed-form delta
2. Pricing-by-integration delta
3. CRR finite-difference delta using a binomial tree construction

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.greeks import eu_bs_delta, eu_crr_delta, delta_int_sn
```

Black-Scholes delta:

```python
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
def vcall_prime(S, K):
    return 1.0 if S > K else 0.0


delta = delta_int_sn(
    t=0,
    f_prime=lambda S: vcall_prime(S, 100),
    St=120,
    T=1,
    r=0.02,
    sigma=0.3,
    a=-float("inf"),
    b=float("inf"),
)
```

---

### `american.py`

Contains valuation routines for American options.

Currently this module includes:

- A perpetual American put option in the Black-Scholes model
- A finite-maturity American call/put option using the CRR binomial tree

```python
from comp_fin_lab.american import perp_am_put, am_crr
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

Available function:

```python
perp_am_put(K, r, sigma, S_min=1e-6, S_max=200, n_grid=200, S_grid=None)
```

The function returns:

```text
S_grid   -> stock-price grid
v_grid   -> perpetual American put value on the grid
x_star   -> optimal exercise boundary
```

---

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

Available function:

```python
am_crr(S_ini, K, T, r, sigma, N, opttype, c=1)
```

where:

```text
opttype = "C"  -> American call
opttype = "P"  -> American put
```

The function returns:

```text
price   -> American option price at time 0
C       -> option value tree
S       -> underlying stock price tree
```

At each node of the tree, the function compares the continuation value with the immediate exercise value:

```text
American value = max(continuation value, exercise payoff)
```

For a non-dividend-paying stock, the American call should be approximately equal to the corresponding European call. The American put can be more valuable than the European put because early exercise may be optimal.

---

### `heston.py`

Contains Heston stochastic volatility path simulation.

The Heston model is defined in continuous time as:

```text
dS(t) = r S(t) dt + sqrt(gamma(t)) S(t) dW_S(t)

dgamma(t) = kappa(theta - gamma(t)) dt
            + sigma sqrt(gamma(t)) dW_gamma(t)
```

with correlation:

```text
corr(dW_S(t), dW_gamma(t)) = rho
```

Here `gamma(t)` is the variance process, and `sqrt(gamma(t))` is the stochastic volatility process.

The module simulates the Heston model on a discrete time grid using:

1. Euler-type discretization for the variance process
2. Log-Euler discretization for the stock price process
3. Cholesky decomposition to generate correlated Brownian shocks

Available functions:

```python
random_number_gen(M, I, seed=None)
heston_paths(S0, r, gamma0, kappa, theta, sigma, rho, T, M, I, seed=None, rand=None)
```

Internal helper functions:

```python
_cholesky_from_rho(rho)
_sde_gamma(gamma0, kappa, theta, sigma, T, M, I, rand, rho)
```

Simulating Heston stock price and variance paths: 

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
```

The returned variance paths can be converted into volatility paths by taking the square root:

```python
V = np.sqrt(gamma)
```

where:

```text
gamma(t)        -> variance process
sqrt(gamma(t)) -> volatility process
```

Parameters:

```text
S0 : float
    Initial stock price.

r : float
    Risk-free interest rate.

gamma0 : float
    Initial variance.

kappa : float
    Speed of mean reversion of the variance process.

theta : float
    Long-run variance level.

sigma : float
    Volatility of variance, also called vol-of-vol.

rho : float
    Correlation between stock and variance Brownian shocks.

T : float
    Time horizon.

M : int
    Number of time steps.

I : int
    Number of simulated paths.

seed : int, optional
    Random seed for reproducibility.

rand : ndarray, optional
    Pre-generated independent standard normal shocks with shape (2, M + 1, I).
    If None, random numbers are generated internally.
```

Returns:

```text
S : ndarray
    Simulated stock price paths with shape (M + 1, I).

gamma : ndarray
    Simulated variance paths with shape (M + 1, I).
```

The stock price is simulated using a log-Euler update:

```text
S(t + dt) = S(t) exp((r - 0.5 gamma(t))dt
                    + sqrt(gamma(t)) dW_S(t))
```

The variance process is simulated using an Euler-type update with non-negativity truncation:

```text
gamma(t + dt) = max(0,
                    gamma(t)
                    + kappa(theta - gamma(t))dt
                    + sigma sqrt(gamma(t)) dW_gamma(t))
```

The truncation ensures that simulated variance values remain non-negative.


Random number generation: 

```python
from comp_fin_lab.heston import random_number_gen

rand = random_number_gen(
    M=500,
    I=10000,
    seed=42,
)
```

This generates independent standard normal shocks with shape:

```text
(2, M + 1, I)
```

The first dimension corresponds to the two Brownian shocks:

```text
row 0 -> stock price shock
row 1 -> variance shock
```

The correlation between the two Brownian motions is imposed later using the Cholesky decomposition of the correlation matrix:

```text
[[1,   rho],
 [rho, 1  ]]
```

---

## 4. Tests and Reproducible Diagnostics

The test files are written as executable Python scripts. They can be run directly.

From the project root:

```bash
cd tests

python test_packaging_works.py
python test_payoffs.py
python test_binom.py
python test_bs.py
python test_barriers.py
python test_integration.py
python test_delta.py
python test_american.py
```

The tests generate diagnostic output and plots under:

```text
tests/plots/
```

---

## 5. Test Rationale

### `test_packaging_works.py`

Checks whether the package can be imported:

```python
import comp_fin_lab

print("Package imported successfully")
```

This verifies that the project is correctly structured as an installable Python package.

---

### `test_payoffs.py`

Checks the basic payoff functions:

```python
vcall(S, K)
vput(S, K)
vopt(S, K, call=True)
vopt(S, K, call=False)
```

It tests both scalar inputs and vectorized array inputs.

The goal is to confirm that payoff functions behave correctly before they are used inside pricing models.

---

### `test_binom.py`

This script performs two main CRR diagnostics.

#### 1. CRR lattice structure

The script computes:

```python
S, V, delta_t = eu_crr_lattice(...)
```

and verifies that:

```python
eu_crr(...) == V[0, 0]
```

It then plots:

- The CRR stock-price lattice
- The CRR call-option value lattice
- One random binomial path through the tree

Generated plot:

```text
tests/plots/eu_crr_stock_and_call_structures_with_one_random_path.png
```

This diagnostic checks that the stock-price tree and backward-inducted option-value tree are internally consistent.

#### 2. Volatility sensitivity and put-call parity

The script varies volatility over a grid:

```python
sigmas = np.linspace(0.01, 5, 500)
```

For each volatility, it computes CRR call and put prices.

It checks:

- Call prices remain within no-arbitrage bounds
- Put prices remain within no-arbitrage bounds
- Put-call parity holds approximately

The put-call parity expression checked is:

```text
P - C - K exp(-rT) + S0 = 0
```

Generated plot:

```text
tests/plots/eu_crr_sigma_sensitivity_parity.png
```

---

### `test_bs.py`

Compares CRR prices against Black-Scholes prices across a range of initial stock prices.

The script varies:

```python
S_arr = np.linspace(20, 300, 1000)
```

For each stock price, it computes:

1. Standard CRR call price
2. Anchored CRR call price
3. Black-Scholes call price

The anchored CRR tree uses:

```python
c = (K / S0) ** (2 / M)
```

The plotted quantities are:

```text
CRR - Black-Scholes
Anchored CRR - Black-Scholes
```

Generated plot:

```text
tests/plots/bs_crr_anchored_error.png
```

This test checks how well the binomial approximation matches the Black-Scholes benchmark and whether anchoring improves the approximation error.

---

### `test_bs_paths.py`

Tests the Black-Scholes stock-price path simulation function `bs_paths`.

The script uses:

```python
S0 = 100
r = 0.05
sigma = 0.3
T = 1

M0 = 500
M = int(M0 * T)
I = 10000

seed = 42
```

It simulates paths using:

```python
S, S_hat, time_grid = bs_paths(
    S0=S0,
    r=r,
    sigma=sigma,
    T=T,
    M=M,
    I=I,
    seed=seed,
)
```

The test checks:

1. The shape of the stock path matrix
2. The shape of the discounted stock path matrix
3. The shape of the time grid
4. That all paths start at `S0`
5. That all discounted paths start at `S0`
6. That all simulated values are finite
7. That stock prices and discounted stock prices remain positive
8. That the discounted stock price is approximately a martingale
9. That the undiscounted stock price satisfies the risk-neutral expectation
10. That terminal log returns have the correct theoretical mean and variance
11. That the simulation is reproducible when the same seed is used

The key martingale check is:

```text
E[S_hat(T)] = S0
```

The test checks this approximately using:

```python
assert abs(S_hat[-1].mean() - S0) < 1.5
```

The undiscounted stock price should satisfy:

```text
E[S(T)] = S0 exp(rT)
```

The test checks this approximately using:

```python
assert abs(S[-1].mean() - S0 * np.exp(r * T)) < 2.0
```

The terminal log returns should satisfy:

```text
log(S(T) / S0) ~ N((r - 0.5 sigma^2)T, sigma^2 T)
```

The test compares the sample mean and variance of terminal log returns against:

```python
theoretical_mean = (r - 0.5 * sigma**2) * T
theoretical_var = sigma**2 * T
```

Generated plots:

```text
plots/bs_stock_and_discounted_paths.png
plots/bs_terminal_log_return_distribution.png
```

The first plot shows simulated Black-Scholes stock price paths and discounted stock price paths.

The second plot shows the terminal log-return distribution, together with the theoretical mean and sample mean.

This test checks whether the simulated Black-Scholes paths have the correct risk-neutral expectation, martingale property, terminal distribution, and reproducibility.

---

### `test_bs_pde.py`

Tests the Black-Scholes PDE solver `bs_pde`.

The script uses:

```python
S0 = 120
K = 100
T = 1
r = 0.02
sigma = 0.3

S_max = max(4 * K, 4 * S0)
S_steps = 300
t_steps = 20000
```

First, the script computes a European call price using the PDE solver:

```python
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
```

It also computes a European put price:

```python
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
```

The PDE prices are compared against analytical Black-Scholes prices from `eu_bs`:

```python
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
```

The test checks:

1. PDE call and put prices are finite
2. Black-Scholes call and put prices are finite
3. PDE prices are non-negative
4. PDE prices are close to the analytical Black-Scholes benchmark
5. The PDE value grids have the correct shape
6. The stock-price grid has the correct shape
7. The time-to-maturity grid has the correct shape
8. All PDE grid values are finite
9. The initial condition at `tau = 0` equals the payoff
10. Boundary conditions are correctly imposed
11. PDE price curves are close to Black-Scholes price curves in the central stock-price region
12. Pricing errors decrease as the stock grid becomes finer

Because the explicit finite-difference PDE method is an approximation, the single-price comparison uses a moderate tolerance:

```python
assert call_error < 0.25
assert put_error < 0.25
```

The curve comparison avoids `S = 0` because the Black-Scholes closed form contains `log(S/K)`.

The central region is defined as:

```python
central_mask = (S_eval >= 0.25 * K) & (S_eval <= 3.0 * K)
```

The mean absolute pricing error in this region is checked using:

```python
assert call_abs_error_curve[central_mask].mean() < 0.20
assert put_abs_error_curve[central_mask].mean() < 0.20
```

The convergence test uses:

```python
S_steps_values = np.array([50, 100, 200, 300])
```

For each stock grid size, the number of time steps is chosen automatically so that the explicit scheme remains stable:

```python
t_steps_j = int(
    max(
        2000,
        np.ceil(3 * T * (sigma**2 * s_steps**2 + abs(r))),
    )
)
```

The test then checks that the finest grid is more accurate than the coarsest grid:

```python
assert call_errors[-1] < call_errors[0]
assert put_errors[-1] < put_errors[0]
```

Generated plots:

```text
tests/plots/bs_pde_vs_black_scholes_prices.png
tests/plots/bs_pde_absolute_pricing_error.png
tests/plots/bs_pde_convergence.png
```

The first plot compares PDE and Black-Scholes price curves for calls and puts.

The second plot shows the absolute pricing error over the stock-price grid.

The third plot shows convergence as the number of stock grid steps increases.

This test checks whether the explicit finite-difference approximation solves the Black-Scholes PDE correctly and whether it converges toward the closed-form Black-Scholes benchmark.

---

### `test_barriers.py`

Prices a down-and-out call for different barrier levels:

```python
B_values = np.linspace(50, 150, 100)
```

For each barrier, it computes:

```python
down_and_out_crr(...)
```

It checks:

- Barrier option values are non-negative
- A down-and-out call is never more expensive than the vanilla call
- The option value decreases as the barrier increases
- If the barrier is above the initial stock price, the option is knocked out immediately

Generated plot:

```text
tests/plots/down_and_out_crr_barrier_sensitivity.png
```

This diagnostic verifies that the barrier option behaves consistently with financial intuition.

---

### `test_integration.py`

This script tests pricing by integration.

#### 1. Integration vs Black-Scholes

It computes a European call price using numerical integration and compares it with the Black-Scholes formula.

The relative absolute error is checked:

```text
|Black-Scholes price - Integration price| / Black-Scholes price
```

This confirms that integration over the standard normal distribution correctly reproduces the Black-Scholes price.

#### 2. Power call: integration vs CRR

The script prices a power call payoff:

```python
pcall(S, K, alpha)
```

using:

1. Pricing by integration
2. CRR binomial pricing

The test allows a tolerance because CRR is a discrete approximation.

#### 3. Runtime comparison

It compares average runtime between integration and CRR for the power call.

Generated plot:

```text
tests/plots/integration_vs_crr_power_call_runtime.png
```

This diagnostic shows that, for a one-dimensional Black-Scholes expectation, numerical integration can be much faster than a large binomial tree.

---

### `test_delta.py`

Compares option deltas from three methods:

1. Black-Scholes closed-form delta
2. Delta by numerical integration
3. CRR delta approximation

For a call, the test checks:

```text
0 <= Delta_call <= 1
```

For a put, the test checks:

```text
-1 <= Delta_put <= 0
```

It also checks put-call delta parity:

```text
Delta_call - Delta_put = 1
```

The integration delta and CRR delta are compared against Black-Scholes delta as the benchmark.

---

### `test_american.py`

1. Tests the perpetual American put implementation.

The function:

```python
perp_am_put(...)
```

returns:

```text
S_grid
v_grid_am
x_star
```

The script compares the perpetual American put value with:

- Immediate exercise payoff
- A finite-maturity European put from Black-Scholes

It checks:

- Values are finite
- Values are non-negative
- The exercise boundary satisfies `0 < x_star < K`
- In the exercise region, value equals payoff
- The American put value is at least the intrinsic value
- The perpetual American put is at least as valuable as the finite-maturity European put
- Put value decreases as stock price increases

Generated plot:

```text
tests/plots/perpetual_american_put.png
```

This diagnostic illustrates the exercise boundary and the value of early exercise.

2. American CRR vs European CRR test

The function:

```python
am_crr(...)
```

returns:

```text
price
C
S
```

where:

```text
price   -> American option price at time 0
C       -> option value tree
S       -> underlying stock price tree
```

The script compares American CRR prices with European CRR prices computed using:

```python
eu_crr(...)
```

It checks:

- American call values are finite
- American put values are finite
- American call price is at least the European call price
- American put price is at least the European put price
- For a non-dividend-paying stock, the American call is approximately equal to the European call
- The returned option-value and stock-price trees have the expected shape

The core financial identity being checked is:

```text
American option value >= European option value
```

For a non-dividend-paying stock:

```text
American call ≈ European call
```

because early exercise of a call is not optimal without dividends.

For puts:

```text
American put >= European put
```

because early exercise may be valuable.

---

3. Convergence over number of binomial steps

The script varies the number of CRR time steps:

```python
N_values = np.array([5, 10, 25, 50, 100, 200, 500])
```

For each `N`, it computes:

- American call price
- European call price
- American put price
- European put price

Generated plot:

```text
tests/plots/american_crr_convergence.png
```

This diagnostic checks whether the American CRR prices stabilize as the binomial tree becomes finer.

---

4. Volatility sensitivity test

The script varies volatility over a grid:

```python
sigmas = np.linspace(0.05, 1.0, 100)
```

For each volatility, it computes:

- American call price
- European call price
- American put price
- European put price

It checks:

- Prices are finite
- Prices are non-negative
- American prices dominate European prices
- American call prices are approximately equal to European call prices in the no-dividend case
- Option prices generally increase with volatility

Generated plot:

```text
tests/plots/american_crr_sigma_sensitivity.png
```

This diagnostic shows how American and European option prices respond to changes in volatility.

---

5. Early exercise premium test

The script computes the early exercise premium:

```text
American price - European price
```

for calls and puts.

Generated plot:

```text
tests/plots/american_crr_early_exercise_premium.png
```

This diagnostic illustrates that:

- The call early-exercise premium is approximately zero for a non-dividend-paying stock
- The put early-exercise premium can be positive because early exercise may be optimal 

---

### `test_heston_paths.py`

Tests the Heston stochastic volatility path simulation function `heston_paths`.

The script uses:

```python
gamma0 = 0.04
kappa = 2
sigma = 0.3
theta = 0.04
rho = -0.9

S0 = 100
r = 0.05
T = 1

M0 = 500
M = int(M0 * T)
I = 10000

seed = 42
```

It simulates Heston paths using:

```python
S, gamma = heston_paths(
    S0=S0,
    r=r,
    gamma0=gamma0,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    rho=rho,
    T=T,
    M=M,
    I=I,
    seed=seed,
)
```

The variance process is converted into volatility using:

```python
V = np.sqrt(gamma)
```

The test checks:

1. The shape of the stock price path matrix
2. The shape of the variance path matrix
3. The shape of the volatility path matrix
4. That all stock paths start at `S0`
5. That all variance paths start at `gamma0`
6. That all volatility paths start at `sqrt(gamma0)`
7. That all simulated stock prices are finite
8. That all simulated variance values are finite
9. That all simulated volatility values are finite
10. That stock prices remain positive under the log-Euler update
11. That variance values remain non-negative because of the truncation scheme
12. That volatility values are non-negative
13. That the simulation is reproducible when the same seed is used

The key initial condition checks are:

```python
assert np.allclose(S[0], S0)
assert np.allclose(gamma[0], gamma0)
assert np.allclose(V[0], np.sqrt(gamma0))
```

The key positivity checks are:

```python
assert np.all(S > 0)
assert np.all(gamma >= 0)
assert np.all(V >= 0)
```

The reproducibility check reruns the simulation with the same seed:

```python
S_same, gamma_same = heston_paths(
    S0=S0,
    r=r,
    gamma0=gamma0,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    rho=rho,
    T=T,
    M=M,
    I=I,
    seed=seed,
)
```

and verifies:

```python
assert np.allclose(S, S_same)
assert np.allclose(gamma, gamma_same)
```

Generated plot:

```text
plots/heston_price_and_volatility_paths.png
```

The plot shows:

1. Simulated Heston stock price paths
2. Simulated Heston volatility paths

This test checks whether the Heston simulation produces valid stock, variance, and volatility paths, preserves the correct initial conditions, enforces non-negative variance, and gives reproducible results when a fixed random seed is used.

---


## 6. Generated Plots

The current diagnostic plots are:

```text
tests/plots/bs_crr_anchored_error.png
tests/plots/down_and_out_crr_barrier_sensitivity.png
tests/plots/eu_crr_sigma_sensitivity_parity.png
tests/plots/eu_crr_stock_and_call_structures_with_one_random_path.png
tests/plots/integration_vs_crr_power_call_runtime.png
tests/plots/perpetual_american_put.png
```

These plots are generated by the test scripts and document the numerical behaviour of the implemented models.

For a clean GitHub repository, there are two possible approaches:

### Option 1: Do not commit generated plots

Keep `tests/plots/` in `.gitignore`.

Users can reproduce the plots by running the tests.

### Option 2: Commit selected plots for README display

Move selected figures to:

```text
docs/plots/
```

and reference them in the README.

Example:

```markdown
![CRR vs Black-Scholes error](docs/plots/bs_crr_anchored_error.png)
```

Recommended approach:

- Keep `tests/plots/` ignored
- Copy only polished portfolio figures to `docs/plots/`

---

## 7. Example Workflow

A typical workflow is:

```python
from comp_fin_lab.payoffs import vcall
from comp_fin_lab.bs import eu_bs
from comp_fin_lab.binom import eu_crr
from comp_fin_lab.greeks import eu_bs_delta

S0 = 120
K = 100
T = 1
r = 0.02
sigma = 0.3
M = 100

bs_price = eu_bs(
    t=0,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
)

crr_price = eu_crr(
    g=vcall,
    S0=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    M=M,
)

delta = eu_bs_delta(
    t=0,
    St=S0,
    K=K,
    T=T,
    r=r,
    sigma=sigma,
    call=1,
)

print("Black-Scholes call price:", bs_price)
print("CRR call price:", crr_price)
print("Black-Scholes call delta:", delta)
```

---

## 8. Mathematical Scope

This package currently focuses on the following modelling ideas:

### Risk-neutral pricing

Most functions use the idea that the option price is a discounted risk-neutral expectation:

```text
V(t) = exp(-r(T-t)) E^Q[payoff]
```

### Binomial approximation

The CRR model approximates the risk-neutral stock process using a recombining binomial tree.

### Black-Scholes benchmark

The Black-Scholes formula is used as the closed-form benchmark for European call and put pricing.

### Pricing by integration

In the Black-Scholes model, terminal stock prices can be written as a function of a standard normal random variable. Therefore, option prices can be computed by integrating over the standard normal density.

### Greeks

Delta is computed using:

- Closed-form Black-Scholes expressions
- Differentiation under the integral sign
- CRR finite-difference approximations

### Optimal stopping

The perpetual American put is treated as a free-boundary problem. Below the exercise boundary, immediate exercise is optimal. Above the boundary, the value solves a stationary Black-Scholes ODE.

---

## 9. Future Extensions

Planned extensions include:

- Monte Carlo option pricing
- Random number generation methods
- Inverse transform sampling
- Acceptance-rejection sampling
- Ornstein-Uhlenbeck process simulation
- CIR process simulation
- GBM simulation
- Heston stochastic volatility pricing
- Characteristic-function based pricing
- Fourier transform pricing
- Laplace transform pricing
- Implied volatility
- Calibration routines
- Backtesting and hedging examples
- Documentation pages and example notebooks

---

## 10. Repository Philosophy

This repository is designed as a reusable computational finance library rather than a collection of isolated notebooks.

The aim is to preserve mathematically verified implementations while turning them into importable, testable, and extensible Python modules.

The package can be used as a foundation for:

- Derivative pricing projects
- Stochastic process simulation
- Model calibration
- Risk analysis
- Delta hedging backtests
- Quant developer portfolio projects