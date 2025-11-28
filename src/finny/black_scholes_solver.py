"""
Black-Scholes PDE Solver using Crank-Nicolson Method

This module implements a numerical solution to the Black-Scholes partial differential equation
using the Crank-Nicolson finite difference method.

The Black-Scholes PDE:
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

Where:
- V(S,t) is the option value
- S is the underlying asset price
- t is time
- σ is volatility
- r is risk-free rate
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
try:
    import seaborn as sns  # type: ignore
    sns.set_palette("husl")
except ImportError:
    sns = None
from typing import Tuple, Callable, Optional, Union


class BlackScholesConfig:
    """Configuration class for Black-Scholes parameters"""
    
    def __init__(self, 
                 S_max: float = 200.0,
                 K: float = 100.0,
                 T: float = 1.0,
                 r: float = 0.05,
                 sigma: float = 0.2,
                 option_type: str = 'call'):
        """
        Initialize Black-Scholes configuration
        
        Parameters:
        -----------
        S_max : float
            Maximum stock price for the grid
        K : float
            Strike price
        T : float
            Time to expiration
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        """
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()


def payoff_function(S: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """
    Calculate the payoff function for European options
    
    Parameters:
    -----------
    S : np.ndarray
        Array of stock prices
    K : float
        Strike price
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    np.ndarray
        Payoff values
    """
    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def boundary_conditions(S: np.ndarray, t: float, config: BlackScholesConfig) -> Tuple[float, float]:
    """
    Calculate boundary conditions for the Black-Scholes PDE
    
    Parameters:
    -----------
    S : np.ndarray
        Stock price grid
    t : float
        Current time
    config : BlackScholesConfig
        Configuration object
        
    Returns:
    --------
    Tuple[float, float]
        Lower and upper boundary values
    """
    time_to_exp = config.T - t
    
    if config.option_type == 'call':
        # For call option: V(0,t) = 0, V(S_max,t) ≈ S_max - K*exp(-r*(T-t))
        lower_bc = 0.0
        upper_bc = config.S_max - config.K * np.exp(-config.r * time_to_exp)
    else:  # put option
        # For put option: V(0,t) = K*exp(-r*(T-t)), V(S_max,t) = 0
        lower_bc = config.K * np.exp(-config.r * time_to_exp)
        upper_bc = 0.0
        
    return lower_bc, upper_bc


def crank_nicolson_solver(config: BlackScholesConfig, 
                         N_S: int = 100, 
                         N_t: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Black-Scholes PDE using the Crank-Nicolson method
    
    Parameters:
    -----------
    config : BlackScholesConfig
        Configuration object containing all parameters
    N_S : int
        Number of space grid points
    N_t : int
        Number of time grid points
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        S_grid, t_grid, V_grid where V_grid[i,j] = V(S_grid[i], t_grid[j])
    """
    
    # Setup grids
    S_grid = np.linspace(0, config.S_max, N_S)
    t_grid = np.linspace(0, config.T, N_t)
    
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]
    
    # Initialize solution matrix
    V = np.zeros((N_S, N_t))
    
    # Set initial condition (payoff at expiry)
    V[:, -1] = payoff_function(S_grid, config.K, config.option_type)
    
    # Coefficients for the finite difference scheme
    # We'll solve for interior points (excluding boundaries)
    interior_indices = np.arange(1, N_S - 1)
    S_interior = S_grid[interior_indices]
    
    # Coefficients for the PDE discretization
    alpha = 0.5 * config.sigma**2 * S_interior**2 / dS**2
    beta = 0.5 * config.r * S_interior / dS
    gamma = config.r
    
    # Build matrices for Crank-Nicolson scheme
    # AV^{n+1} = BV^n + boundary terms
    
    # Matrix A (implicit part)
    A_diag = 1 + dt * (alpha + 0.5 * gamma)
    A_upper = -0.5 * dt * (alpha + beta)
    A_lower = -0.5 * dt * (alpha - beta)
    
    A = diags([A_lower[1:], A_diag, A_upper[:-1]], 
              offsets=(-1, 0, 1),  # type: ignore
              shape=(N_S-2, N_S-2), 
              format='csr')
    
    # Matrix B (explicit part)
    B_diag = 1 - dt * (alpha + 0.5 * gamma)
    B_upper = 0.5 * dt * (alpha + beta)
    B_lower = 0.5 * dt * (alpha - beta)
    
    B = diags([B_lower[1:], B_diag, B_upper[:-1]], 
              offsets=(-1, 0, 1),  # type: ignore
              shape=(N_S-2, N_S-2), 
              format='csr')
    
    # Time stepping (backward in time)
    for j in range(N_t - 2, -1, -1):
        # Get boundary conditions
        lower_bc, upper_bc = boundary_conditions(S_grid, t_grid[j], config)
        V[0, j] = lower_bc
        V[-1, j] = upper_bc
        
        # Right hand side
        rhs = B @ V[1:-1, j+1]
        
        # Add boundary condition contributions
        # From lower boundary
        rhs[0] += 0.5 * dt * (alpha[0] - beta[0]) * (V[0, j+1] + V[0, j])
        # From upper boundary  
        rhs[-1] += 0.5 * dt * (alpha[-1] + beta[-1]) * (V[-1, j+1] + V[-1, j])
        
        # Solve the linear system
        V[1:-1, j] = np.array(spsolve(A, rhs))
    
    return S_grid, t_grid, V


def analytical_black_scholes(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = 'call') -> float:
    """
    Calculate the analytical Black-Scholes option price
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration
    r : float
        Risk-free rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def plot_results(S_grid: np.ndarray, t_grid: np.ndarray, V_grid: np.ndarray, 
                config: BlackScholesConfig, comparison_times: Optional[list] = None):
    """
    Plot the results of the Black-Scholes PDE solution
    
    Parameters:
    -----------
    S_grid : np.ndarray
        Stock price grid
    t_grid : np.ndarray
        Time grid
    V_grid : np.ndarray
        Solution grid
    config : BlackScholesConfig
        Configuration object
    comparison_times : list, optional
        Times at which to compare with analytical solution
    """
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 3D Surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    S_mesh, T_mesh = np.meshgrid(S_grid, t_grid)
    surf = ax1.plot_surface(S_mesh, T_mesh, V_grid.T, cmap='viridis', alpha=0.8)  # type: ignore
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel('Option Value (V)')  # type: ignore
    ax1.set_title(f'Black-Scholes {config.option_type.title()} Option Value')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2. Heatmap
    ax2 = axes[0, 1]
    im = ax2.imshow(V_grid.T, aspect='auto', origin='lower', cmap='plasma',
                    extent=[S_grid[0], S_grid[-1], t_grid[0], t_grid[-1]])
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Time (t)')
    ax2.set_title('Option Value Heatmap')
    plt.colorbar(im, ax=ax2)
    
    # 3. Option value at different times
    ax3 = axes[1, 0]
    if comparison_times is None:
        comparison_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for t in comparison_times:
        t_idx = np.argmin(np.abs(t_grid - t))
        label = f't = {t:.2f}'
        ax3.plot(S_grid, V_grid[:, t_idx], label=label, linewidth=2)
        
        # Add analytical solution for comparison if at t=0
        if t == 0.0:
            analytical_values = [analytical_black_scholes(s, config.K, config.T, 
                                                        config.r, config.sigma, 
                                                        config.option_type) 
                               for s in S_grid[1:-1]]  # Exclude boundaries
            ax3.plot(S_grid[1:-1], analytical_values, 'k--', 
                    label='Analytical (t=0)', linewidth=2, alpha=0.7)
    
    ax3.set_xlabel('Stock Price (S)')
    ax3.set_ylabel('Option Value (V)')
    ax3.set_title('Option Value vs Stock Price')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Greeks (Delta) at t=0
    ax4 = axes[1, 1]
    delta_numerical = np.gradient(V_grid[:, 0], S_grid)
    ax4.plot(S_grid, delta_numerical, 'b-', label='Numerical Delta', linewidth=2)
    
    # Analytical delta for comparison
    from scipy.stats import norm
    analytical_delta = []
    for s in S_grid:
        if s > 0:
            d1 = (np.log(s / config.K) + (config.r + 0.5 * config.sigma**2) * config.T) / (config.sigma * np.sqrt(config.T))
            if config.option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
        else:
            delta = 0 if config.option_type == 'call' else -1
        analytical_delta.append(delta)
    
    ax4.plot(S_grid, analytical_delta, 'r--', label='Analytical Delta', 
            linewidth=2, alpha=0.7)
    ax4.set_xlabel('Stock Price (S)')
    ax4.set_ylabel('Delta')
    ax4.set_title('Option Delta at t=0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_solver():
    """
    Demonstrate the Black-Scholes solver with example parameters
    """
    print("Black-Scholes PDE Solver using Crank-Nicolson Method")
    print("=" * 55)
    
    # Configuration
    config = BlackScholesConfig(
        S_max=200.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type='call'
    )
    
    print(f"Parameters:")
    print(f"  Strike Price (K): {config.K}")
    print(f"  Time to Expiration (T): {config.T}")
    print(f"  Risk-free Rate (r): {config.r}")
    print(f"  Volatility (σ): {config.sigma}")
    print(f"  Option Type: {config.option_type}")
    print(f"  Max Stock Price: {config.S_max}")
    
    # Solve the PDE
    print("\nSolving Black-Scholes PDE...")
    S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=500)
    
    # Compare with analytical solution at S=100, t=0
    analytical_price = analytical_black_scholes(100, config.K, config.T, 
                                              config.r, config.sigma, 
                                              config.option_type)
    
    # Find numerical price at S=100, t=0
    S_idx = np.argmin(np.abs(S_grid - 100))
    numerical_price = V_grid[S_idx, 0]
    
    print(f"\nResults at S=100, t=0:")
    print(f"  Analytical Price: {analytical_price:.6f}")
    print(f"  Numerical Price:  {numerical_price:.6f}")
    print(f"  Error: {abs(analytical_price - numerical_price):.6f}")
    print(f"  Relative Error: {abs(analytical_price - numerical_price)/analytical_price*100:.4f}%")
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(S_grid, t_grid, V_grid, config)
    
    return S_grid, t_grid, V_grid, config


if __name__ == "__main__":
    # Run demonstration
    S_grid, t_grid, V_grid, config = demonstrate_solver()
