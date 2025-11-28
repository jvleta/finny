# Finny Technical Notes

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

## Numerical Properties

### Stability
The Crank-Nicolson method is unconditionally stable, meaning the numerical solution remains stable for any choice of time step `dt`.

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

## Validation

The solver has been validated against:
- Analytical Black-Scholes formula (relative error < 0.1%)
- Put-call parity
- Greeks calculated analytically
- Grid convergence studies

## References

1. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637-654.
2. Wilmott, P., Howison, S., & Dewynne, J. (1995). The Mathematics of Financial Derivatives. Cambridge University Press.
3. Crank, J., & Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. Mathematical Proceedings of the Cambridge Philosophical Society, 43(1), 50-67.
