# Finny

A financial modeling toolkit for obtaining and visualizing solutions to the Black-Scholes PDE

## Features

- **Numerical PDE Solver**: Implements the Crank-Nicolson scheme for solving the Black-Scholes PDE
- **High Accuracy**: Second-order accurate in both time and space
- **Unconditional Stability**: The Crank-Nicolson method is unconditionally stable
- **Option Types**: Supports both European call and put options
- **Greeks Calculation**: Computes Delta, Gamma, and Theta using finite differences
- **Comprehensive Visualization**: 3D surface plots, contour plots, and time evolution analysis
- **Analytical Comparison**: Includes analytical Black-Scholes formula for validation
- **Interactive Jupyter Notebook**: Step-by-step implementation and examples

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd finny
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install the package locally (editable mode):
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from finny import BlackScholesConfig, crank_nicolson_solver

# Configure option parameters
config = BlackScholesConfig(
    S_max=200.0,    # Maximum stock price for grid
    K=100.0,        # Strike price
    T=1.0,          # Time to expiration (years)
    r=0.05,         # Risk-free rate
    sigma=0.2,      # Volatility
    option_type='call'
)

# Solve the PDE
S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=1000)

# V_grid[i,j] contains the option value at S_grid[i] and t_grid[j]
print(f"Option price at S=100, t=0: ${V_grid[50, 0]:.4f}")
```

### Running Examples

Execute the comprehensive examples:

```bash
python -m finny.examples
```

This will run five different examples demonstrating:
1. Basic call option pricing
2. Put options with different volatilities
3. Effect of time to expiration
4. Greeks analysis
5. Parameter sensitivity analysis

## API Reference

### BlackScholesConfig

Configuration class for Black-Scholes parameters:

```python
config = BlackScholesConfig(
    S_max=200.0,           # Maximum stock price
    K=100.0,               # Strike price
    T=1.0,                 # Time to expiration
    r=0.05,                # Risk-free rate
    sigma=0.2,             # Volatility
    option_type='call'     # 'call' or 'put'
)
```

### crank_nicolson_solver

Main solver function:

```python
S_grid, t_grid, V_grid = crank_nicolson_solver(
    config,     # BlackScholesConfig object
    N_S=100,    # Number of stock price grid points
    N_t=1000    # Number of time grid points
)
```

**Returns:**
- `S_grid`: Stock price grid points
- `t_grid`: Time grid points  
- `V_grid`: Option value grid (S_grid.shape[0] × t_grid.shape[0])

### analytical_black_scholes

Analytical Black-Scholes formula for comparison:

```python
price = analytical_black_scholes(
    S=100.0,           # Current stock price
    K=100.0,           # Strike price
    T=1.0,             # Time to expiration
    r=0.05,            # Risk-free rate
    sigma=0.2,         # Volatility
    option_type='call' # 'call' or 'put'
)
```

### plot_results

Comprehensive plotting function:

```python
plot_results(S_grid, t_grid, V_grid, config, comparison_times=[0.0, 0.5, 1.0])
```

Generates:
- 3D surface plot of option values
- Heatmap visualization
- Option value evolution over time
- Greeks (Delta) comparison with analytical solution

## Examples

### Example 1: Basic Call Option

```python
from finny import *

config = BlackScholesConfig(K=100, T=0.5, r=0.03, sigma=0.15, option_type='call')
S_grid, t_grid, V_grid = crank_nicolson_solver(config)

# Price at-the-money option at t=0
atm_idx = np.argmin(np.abs(S_grid - 100))
numerical_price = V_grid[atm_idx, 0]
analytical_price = analytical_black_scholes(100, 100, 0.5, 0.03, 0.15, 'call')

print(f"Numerical:  ${numerical_price:.6f}")
print(f"Analytical: ${analytical_price:.6f}")
print(f"Error:      ${abs(numerical_price - analytical_price):.6f}")
```

### Example 2: Greeks Calculation

```python
# Calculate Delta using finite differences
dS = S_grid[1] - S_grid[0]
delta = np.gradient(V_grid, dS, axis=0)

# Calculate Gamma (second derivative)
gamma = np.gradient(delta, dS, axis=0)

# Plot Greeks at t=0
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(S_grid, delta[:, 0])
plt.title('Delta')

plt.subplot(1, 2, 2)
plt.plot(S_grid, gamma[:, 0])
plt.title('Gamma')
```

### Example 3: Volatility Sensitivity

```python
volatilities = [0.1, 0.2, 0.3, 0.4]
prices = []

for vol in volatilities:
    config = BlackScholesConfig(K=100, T=1.0, r=0.05, sigma=vol, option_type='call')
    S_grid, t_grid, V_grid = crank_nicolson_solver(config)
    
    # ATM price at t=0
    atm_idx = np.argmin(np.abs(S_grid - 100))
    prices.append(V_grid[atm_idx, 0])

plt.plot(volatilities, prices, 'o-')
plt.xlabel('Volatility')
plt.ylabel('Option Price')
plt.title('Volatility Sensitivity')
```

## Numerical Properties
See `TECHNICAL_NOTES.md` for mathematical background, boundary conditions, and numerical properties.

## Performance Tips

1. **Grid Size**: Start with N_S=100, N_t=1000 for good accuracy/speed balance
2. **S_max Selection**: Choose S_max ≈ 2-3 times the strike price
3. **Memory Usage**: Memory requirement is O(N_S × N_t)
4. **Sparse Matrices**: The implementation uses scipy.sparse for efficient matrix operations

## License

Packaging metadata currently marks the project as UNLICENSED. Add a license file and update pyproject.toml if you intend to publish under an open-source license.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue in the repository.
