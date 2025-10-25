# Black-Scholes PDE Solver using Crank-Nicolson Method

A comprehensive Python implementation of the Black-Scholes partial differential equation solver using the Crank-Nicolson finite difference method. This numerical approach provides highly accurate option pricing and Greeks calculation with excellent stability properties.

## Features

- **Numerical PDE Solver**: Implements the Crank-Nicolson scheme for solving the Black-Scholes PDE
- **High Accuracy**: Second-order accurate in both time and space
- **Unconditional Stability**: The Crank-Nicolson method is unconditionally stable
- **Option Types**: Supports both European call and put options
- **Greeks Calculation**: Computes Delta, Gamma, and Theta using finite differences
- **Comprehensive Visualization**: 3D surface plots, contour plots, and time evolution analysis
- **Analytical Comparison**: Includes analytical Black-Scholes formula for validation
- **Interactive Jupyter Notebook**: Step-by-step implementation and examples

## Mathematical Background

The Black-Scholes PDE is given by:

```
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
```

Where:
- `V(S,t)` is the option value
- `S` is the underlying asset price
- `t` is time
- `σ` is volatility
- `r` is the risk-free interest rate

The Crank-Nicolson method discretizes this PDE using an implicit finite difference scheme that averages the forward and backward Euler methods, providing superior stability and accuracy.

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

## Quick Start

### Basic Usage

```python
from black_scholes_solver import BlackScholesConfig, crank_nicolson_solver

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

```python
python examples.py
```

This will run five different examples demonstrating:
1. Basic call option pricing
2. Put options with different volatilities
3. Effect of time to expiration
4. Greeks analysis
5. Parameter sensitivity analysis

### Jupyter Notebook

Open the interactive notebook for step-by-step implementation:

```bash
jupyter notebook black_scholes_notebook.ipynb
```

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
from black_scholes_solver import *

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

### Stability
The Crank-Nicolson method is unconditionally stable, meaning the numerical solution remains stable for any choice of time step `dt`. This is a significant advantage over explicit methods.

### Accuracy
- **Time discretization**: Second-order accurate (O(dt²))
- **Space discretization**: Second-order accurate (O(dS²))
- **Overall accuracy**: O(dt² + dS²)

### Grid Convergence
The solution converges to the analytical Black-Scholes price as the grid is refined. Typical relative errors are less than 0.1% with moderate grid sizes (N_S=100, N_t=1000).

## Boundary Conditions

The solver implements appropriate boundary conditions:

**For Call Options:**
- `V(0,t) = 0` (worthless when S=0)
- `V(S_max,t) ≈ S_max - K×exp(-r×(T-t))` (deep in-the-money)

**For Put Options:**
- `V(0,t) = K×exp(-r×(T-t))` (maximum value when S=0)
- `V(S_max,t) = 0` (worthless when S >> K)

## Performance Tips

1. **Grid Size**: Start with N_S=100, N_t=1000 for good accuracy/speed balance
2. **S_max Selection**: Choose S_max ≈ 2-3 times the strike price
3. **Memory Usage**: Memory requirement is O(N_S × N_t)
4. **Sparse Matrices**: The implementation uses scipy.sparse for efficient matrix operations

## Validation

The solver has been validated against:
- Analytical Black-Scholes formula (relative error < 0.1%)
- Put-call parity
- Greeks calculated analytically
- Grid convergence studies

## File Structure

```
finny/
├── black_scholes_solver.py      # Main solver implementation
├── black_scholes_notebook.ipynb # Interactive Jupyter notebook
├── examples.py                  # Comprehensive examples
├── requirements.txt             # Package dependencies
└── README.md                   # This file
```

## Dependencies

- **numpy**: Numerical computations and array operations
- **scipy**: Sparse matrix operations and linear algebra
- **matplotlib**: Plotting and visualization
- **seaborn**: Enhanced plotting styles
- **jupyter**: Interactive notebook support

## References

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.
2. Wilmott, P., Howison, S., & Dewynne, J. (1995). The Mathematics of Financial Derivatives. Cambridge University Press.
3. Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. Mathematical Proceedings of the Cambridge Philosophical Society, 43(1), 50-67.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue in the repository.
