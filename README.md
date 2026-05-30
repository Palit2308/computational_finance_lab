# Computational Finance Lab

A modular Python project for computational finance and stochastic modelling.


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
- Perpetual American put valuation using an ODE/free-boundary approach
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
from comp_fin_lab.american import perp_am_put
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

Contains valuation of a perpetual American put option in the Black-Scholes model.

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

Tests the perpetual American put implementation.

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