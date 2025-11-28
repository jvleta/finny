"""
Dividend-Enhanced Black-Scholes PDE Solver using Crank-Nicolson Method

This module extends the European Black-Scholes solver to handle dividend-paying stocks.
It supports both continuous dividend yields and discrete dividend payments.

The Dividend-Adjusted Black-Scholes PDE:
∂V/∂t + (1/2)σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0

Where:
- V(S,t) is the option value
- S is the underlying asset price
- t is time
- σ is volatility
- r is risk-free rate
- q is continuous dividend yield

For discrete dividends, the stock price is adjusted at dividend dates.
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
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DividendEvent:
    """Class to represent a discrete dividend payment"""
    time: float  # Time to dividend (in years from now)
    amount: float  # Dividend amount per share
    
    def __post_init__(self):
        if self.time < 0:
            raise ValueError("Dividend time must be non-negative")
        if self.amount < 0:
            raise ValueError("Dividend amount must be non-negative")


class DividendBlackScholesConfig:
    """Configuration class for Dividend-Enhanced Black-Scholes parameters"""
    
    def __init__(self, 
                 S_max: float = 200.0,
                 K: float = 100.0,
                 T: float = 1.0,
                 r: float = 0.05,
                 sigma: float = 0.2,
                 option_type: str = 'call',
                 dividend_yield: float = 0.0,
                 discrete_dividends: Optional[List[DividendEvent]] = None):
        """
        Initialize Dividend Black-Scholes configuration
        
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
        dividend_yield : float
            Continuous dividend yield (q)
        discrete_dividends : List[DividendEvent], optional
            List of discrete dividend payments
        """
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.dividend_yield = dividend_yield
        self.discrete_dividends = discrete_dividends or []
        
        # Validate discrete dividends
        for div in self.discrete_dividends:
            if div.time > self.T:
                raise ValueError(f"Dividend at time {div.time} is after expiration {self.T}")
        
        # Sort dividends by time
        self.discrete_dividends.sort(key=lambda x: x.time)


def payoff_function_dividend(S: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """
    Calculate the payoff function for dividend-paying options
    
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


def boundary_conditions_dividend(S: np.ndarray, t: float, config: DividendBlackScholesConfig) -> Tuple[float, float]:
    """
    Calculate boundary conditions for dividend-paying stock options
    
    Parameters:
    -----------
    S : np.ndarray
        Stock price grid
    t : float
        Current time
    config : DividendBlackScholesConfig
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
        # Account for dividends in upper boundary
        pv_dividends = sum(div.amount * np.exp(-config.r * (div.time - t)) 
                          for div in config.discrete_dividends if div.time > t)
        upper_bc = (config.S_max - pv_dividends) - config.K * np.exp(-config.r * time_to_exp)
        upper_bc = max(upper_bc, 0)  # Ensure non-negative
    else:  # put option
        # For put option: V(0,t) = K*exp(-r*(T-t)), V(S_max,t) = 0
        lower_bc = config.K * np.exp(-config.r * time_to_exp)
        upper_bc = 0.0
        
    return lower_bc, upper_bc


def adjust_stock_price_for_dividends(S_grid: np.ndarray, dividend_amount: float) -> np.ndarray:
    """
    Adjust stock prices for discrete dividend payment
    
    On ex-dividend date, stock price drops by dividend amount.
    This function implements the price adjustment.
    
    Parameters:
    -----------
    S_grid : np.ndarray
        Stock price grid before dividend
    dividend_amount : float
        Dividend amount per share
        
    Returns:
    --------
    np.ndarray
        Adjusted stock price grid
    """
    return np.maximum(S_grid - dividend_amount, 0)


def interpolate_option_values(S_old: np.ndarray, V_old: np.ndarray, 
                            S_new: np.ndarray) -> np.ndarray:
    """
    Interpolate option values onto new stock price grid after dividend adjustment
    
    Parameters:
    -----------
    S_old : np.ndarray
        Old stock price grid
    V_old : np.ndarray
        Option values on old grid
    S_new : np.ndarray
        New stock price grid after dividend adjustment
        
    Returns:
    --------
    np.ndarray
        Interpolated option values
    """
    return np.interp(S_new, S_old, V_old, left=V_old[0], right=V_old[-1])


def dividend_crank_nicolson_solver(config: DividendBlackScholesConfig, 
                                 N_S: int = 100, 
                                 N_t: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Solve the Dividend-Enhanced Black-Scholes PDE using the Crank-Nicolson method
    
    Parameters:
    -----------
    config : DividendBlackScholesConfig
        Configuration object containing all parameters
    N_S : int
        Number of space grid points
    N_t : int
        Number of time grid points
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]
        S_grid, t_grid, V_grid, dividend_indices
    """
    
    # Setup grids
    S_grid = np.linspace(0, config.S_max, N_S)
    t_grid = np.linspace(0, config.T, N_t)
    
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]
    
    # Initialize solution matrix
    V = np.zeros((N_S, N_t))
    
    # Track dividend payment indices
    dividend_indices = []
    dividend_times = [div.time for div in config.discrete_dividends]
    
    for div_time in dividend_times:
        div_idx = np.argmin(np.abs(t_grid - div_time))
        dividend_indices.append(div_idx)
    
    # Set initial condition (payoff at expiry)
    V[:, -1] = payoff_function_dividend(S_grid, config.K, config.option_type)
    
    # Coefficients for the finite difference scheme
    interior_indices = np.arange(1, N_S - 1)
    S_interior = S_grid[interior_indices]
    
    print(f"Solving Dividend-Enhanced {config.option_type} option...")
    print(f"  Continuous dividend yield: {config.dividend_yield:.3f}")
    print(f"  Discrete dividends: {len(config.discrete_dividends)}")
    
    # Time stepping (backward in time)
    for j in range(N_t - 2, -1, -1):
        current_time = t_grid[j]
        
        # Check if we're at a dividend payment date
        is_dividend_date = j in dividend_indices
        dividend_amount = 0.0
        
        if is_dividend_date:
            # Find the dividend amount for this date
            for div_idx, div in enumerate(config.discrete_dividends):
                if abs(div.time - current_time) < dt/2:  # Within half a time step
                    dividend_amount = div.amount
                    print(f"  Processing dividend of ${dividend_amount:.4f} at t={current_time:.4f}")
                    break
        
        # Build coefficient matrices with dividend yield
        # Modified drift term: (r - q) instead of r
        effective_drift = config.r - config.dividend_yield
        
        alpha = 0.5 * config.sigma**2 * S_interior**2 / dS**2
        beta = 0.5 * effective_drift * S_interior / dS
        gamma = config.r
        
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
        
        # Handle discrete dividend if present
        if is_dividend_date and dividend_amount > 0:
            # Adjust stock prices for dividend
            S_grid_adjusted = adjust_stock_price_for_dividends(S_grid, dividend_amount)
            
            # Interpolate option values onto new grid
            V[:, j+1] = interpolate_option_values(S_grid, V[:, j+1], S_grid_adjusted)
            
            # Update interior grid for this time step
            S_interior = S_grid_adjusted[interior_indices]
            
            # Recalculate coefficients with adjusted prices
            alpha = 0.5 * config.sigma**2 * S_interior**2 / dS**2
            beta = 0.5 * effective_drift * S_interior / dS
            
            # Rebuild matrices
            A_diag = 1 + dt * (alpha + 0.5 * gamma)
            A_upper = -0.5 * dt * (alpha + beta)
            A_lower = -0.5 * dt * (alpha - beta)
            
            A = diags([A_lower[1:], A_diag, A_upper[:-1]], 
                      offsets=(-1, 0, 1),  # type: ignore
                      shape=(N_S-2, N_S-2), 
                      format='csr')
            
            B_diag = 1 - dt * (alpha + 0.5 * gamma)
            B_upper = 0.5 * dt * (alpha + beta)
            B_lower = 0.5 * dt * (alpha - beta)
            
            B = diags([B_lower[1:], B_diag, B_upper[:-1]], 
                      offsets=(-1, 0, 1),  # type: ignore
                      shape=(N_S-2, N_S-2), 
                      format='csr')
        
        # Get boundary conditions
        lower_bc, upper_bc = boundary_conditions_dividend(S_grid, current_time, config)
        V[0, j] = lower_bc
        V[-1, j] = upper_bc
        
        # Right hand side
        rhs = B @ V[1:-1, j+1]
        
        # Add boundary condition contributions
        rhs[0] += 0.5 * dt * (alpha[0] - beta[0]) * (V[0, j+1] + V[0, j])
        rhs[-1] += 0.5 * dt * (alpha[-1] + beta[-1]) * (V[-1, j+1] + V[-1, j])
        
        # Solve the linear system
        V[1:-1, j] = np.array(spsolve(A, rhs))
    
    return S_grid, t_grid, V, dividend_indices


def analytical_dividend_black_scholes(S: float, K: float, T: float, r: float, 
                                    sigma: float, dividend_yield: float = 0.0,
                                    discrete_dividends: Optional[List[DividendEvent]] = None,
                                    option_type: str = 'call') -> float:
    """
    Calculate the analytical Black-Scholes option price with dividends
    
    For continuous dividends, we use the modified Black-Scholes formula.
    For discrete dividends, we adjust the stock price by present value of dividends.
    
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
    dividend_yield : float
        Continuous dividend yield
    discrete_dividends : List[DividendEvent], optional
        List of discrete dividend payments
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    from scipy.stats import norm
    
    discrete_dividends = discrete_dividends or []
    
    # Adjust stock price for present value of discrete dividends
    pv_dividends = sum(div.amount * np.exp(-r * div.time) for div in discrete_dividends)
    S_adjusted = S - pv_dividends
    
    # Use dividend-adjusted Black-Scholes formula
    d1 = (np.log(S_adjusted / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S_adjusted * np.exp(-dividend_yield * T) * norm.cdf(d1) - 
                K * np.exp(-r * T) * norm.cdf(d2))
    else:  # put
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                S_adjusted * np.exp(-dividend_yield * T) * norm.cdf(-d1))
    
    return price


def plot_dividend_results(S_grid: np.ndarray, t_grid: np.ndarray, V_grid: np.ndarray, 
                        config: DividendBlackScholesConfig, dividend_indices: List[int],
                        comparison_times: Optional[list] = None):
    """
    Plot the results of the dividend-enhanced Black-Scholes solution
    
    Parameters:
    -----------
    S_grid : np.ndarray
        Stock price grid
    t_grid : np.ndarray
        Time grid
    V_grid : np.ndarray
        Solution grid
    config : DividendBlackScholesConfig
        Configuration object
    dividend_indices : List[int]
        Time indices where dividends are paid
    comparison_times : list, optional
        Times at which to compare with analytical solution
    """
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 3D Surface plot
    ax1 = fig.add_subplot(231, projection='3d')
    S_mesh, T_mesh = np.meshgrid(S_grid, t_grid)
    surf = ax1.plot_surface(S_mesh, T_mesh, V_grid.T, cmap='viridis', alpha=0.8)  # type: ignore
    ax1.set_xlabel('Stock Price (S)')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel('Option Value (V)')  # type: ignore
    ax1.set_title(f'Dividend-Enhanced {config.option_type.title()} Option Value')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2. Heatmap with dividend dates
    ax2 = axes[0, 1]
    im = ax2.imshow(V_grid.T, aspect='auto', origin='lower', cmap='plasma',
                    extent=[S_grid[0], S_grid[-1], t_grid[0], t_grid[-1]])
    
    # Mark dividend payment dates
    for div_idx in dividend_indices:
        ax2.axhline(y=t_grid[div_idx], color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Time (t)')
    ax2.set_title('Option Value with Dividend Dates (white lines)')
    plt.colorbar(im, ax=ax2)
    
    # 3. Dividend impact analysis
    ax3 = axes[0, 2]
    if len(dividend_indices) > 0:
        # Show option value just before and after first dividend
        div_idx = dividend_indices[0]
        if div_idx < len(t_grid) - 1:
            ax3.plot(S_grid, V_grid[:, div_idx+1], 'b-', linewidth=2, 
                    label=f'Before dividend (t={t_grid[div_idx+1]:.3f})')
            ax3.plot(S_grid, V_grid[:, div_idx], 'r-', linewidth=2, 
                    label=f'After dividend (t={t_grid[div_idx]:.3f})')
            ax3.legend()
            ax3.set_title('Dividend Impact on Option Value')
        else:
            ax3.text(0.5, 0.5, 'No dividend impact to show', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Dividend Timeline')
    else:
        ax3.text(0.5, 0.5, 'No discrete dividends', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_title('No Discrete Dividends')
    
    ax3.set_xlabel('Stock Price (S)')
    ax3.set_ylabel('Option Value (V)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Option value at different times
    ax4 = axes[1, 0]
    if comparison_times is None:
        comparison_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for t in comparison_times:
        t_idx = np.argmin(np.abs(t_grid - t))
        label = f't = {t:.2f}'
        ax4.plot(S_grid, V_grid[:, t_idx], label=label, linewidth=2)
        
        # Add analytical solution for comparison if at t=0
        if t == 0.0:
            analytical_values = [analytical_dividend_black_scholes(
                s, config.K, config.T, config.r, config.sigma, 
                config.dividend_yield, config.discrete_dividends, config.option_type) 
                               for s in S_grid[1:-1]]  # Exclude boundaries
            ax4.plot(S_grid[1:-1], analytical_values, 'k--', 
                    label='Analytical (t=0)', linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('Stock Price (S)')
    ax4.set_ylabel('Option Value (V)')
    ax4.set_title('Option Value vs Stock Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Dividend yield impact comparison
    ax5 = axes[1, 1]
    if config.dividend_yield > 0:
        # Compare with zero dividend yield
        S_test = S_grid[len(S_grid)//2]  # Middle stock price
        div_yields = np.linspace(0, config.dividend_yield * 2, 20)
        option_values = []
        
        for q in div_yields:
            val = analytical_dividend_black_scholes(
                S_test, config.K, config.T, config.r, config.sigma, 
                q, config.discrete_dividends, config.option_type)
            option_values.append(val)
        
        ax5.plot(div_yields * 100, option_values, 'g-', linewidth=2)
        ax5.axvline(x=config.dividend_yield * 100, color='r', linestyle='--', 
                   label=f'Current yield: {config.dividend_yield*100:.1f}%')
        ax5.set_xlabel('Dividend Yield (%)')
        ax5.set_ylabel('Option Value ($)')
        ax5.set_title(f'Dividend Yield Impact (S=${S_test:.0f})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No continuous dividend yield', 
                transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Continuous Dividend Analysis')
    
    # 6. Greeks (Delta) comparison
    ax6 = axes[1, 2]
    delta_numerical = np.gradient(V_grid[:, 0], S_grid)
    ax6.plot(S_grid, delta_numerical, 'b-', label='Numerical Delta', linewidth=2)
    
    # Analytical delta for comparison
    from scipy.stats import norm
    analytical_delta = []
    for s in S_grid:
        if s > 0:
            pv_dividends = sum(div.amount * np.exp(-config.r * div.time) 
                             for div in config.discrete_dividends)
            s_adj = s - pv_dividends
            
            if s_adj > 0:
                d1 = (np.log(s_adj / config.K) + 
                     (config.r - config.dividend_yield + 0.5 * config.sigma**2) * config.T) / \
                     (config.sigma * np.sqrt(config.T))
                
                if config.option_type == 'call':
                    delta = np.exp(-config.dividend_yield * config.T) * norm.cdf(d1)
                else:
                    delta = np.exp(-config.dividend_yield * config.T) * (norm.cdf(d1) - 1)
            else:
                delta = 0 if config.option_type == 'call' else -1
        else:
            delta = 0 if config.option_type == 'call' else -1
        analytical_delta.append(delta)
    
    ax6.plot(S_grid, analytical_delta, 'r--', label='Analytical Delta', 
            linewidth=2, alpha=0.7)
    ax6.set_xlabel('Stock Price (S)')
    ax6.set_ylabel('Delta')
    ax6.set_title('Option Delta at t=0')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_dividend_solver():
    """
    Demonstrate the dividend-enhanced Black-Scholes solver
    """
    print("Dividend-Enhanced Black-Scholes PDE Solver")
    print("=" * 60)
    
    # Example 1: Continuous dividend yield
    print("\n1. Continuous Dividend Yield Example:")
    print("-" * 40)
    
    config1 = DividendBlackScholesConfig(
        S_max=200.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type='call',
        dividend_yield=0.03  # 3% continuous dividend yield
    )
    
    print(f"Parameters:")
    print(f"  Strike Price (K): ${config1.K}")
    print(f"  Time to Expiration (T): {config1.T}")
    print(f"  Risk-free Rate (r): {config1.r}")
    print(f"  Volatility (σ): {config1.sigma}")
    print(f"  Dividend Yield (q): {config1.dividend_yield}")
    print(f"  Option Type: {config1.option_type}")
    
    # Solve with dividends
    S_grid1, t_grid1, V_grid1, div_indices1 = dividend_crank_nicolson_solver(
        config1, N_S=100, N_t=500
    )
    
    # Compare with analytical solution
    S_test = 100.0
    analytical_price1 = analytical_dividend_black_scholes(
        S_test, config1.K, config1.T, config1.r, config1.sigma, 
        config1.dividend_yield, config1.discrete_dividends, config1.option_type
    )
    
    S_idx1 = np.argmin(np.abs(S_grid1 - S_test))
    numerical_price1 = V_grid1[S_idx1, 0]
    
    print(f"\nResults at S=${S_test}, t=0:")
    print(f"  Analytical Price: ${analytical_price1:.6f}")
    print(f"  Numerical Price:  ${numerical_price1:.6f}")
    print(f"  Error: ${abs(analytical_price1 - numerical_price1):.6f}")
    
    # Example 2: Discrete dividends
    print("\n\n2. Discrete Dividend Example:")
    print("-" * 40)
    
    # Create discrete dividend schedule (quarterly dividends)
    dividends = [
        DividendEvent(time=0.25, amount=1.0),  # $1 dividend in 3 months
        DividendEvent(time=0.50, amount=1.0),  # $1 dividend in 6 months
        DividendEvent(time=0.75, amount=1.0),  # $1 dividend in 9 months
    ]
    
    config2 = DividendBlackScholesConfig(
        S_max=200.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type='put',
        dividend_yield=0.0,  # No continuous yield
        discrete_dividends=dividends
    )
    
    print(f"Parameters:")
    print(f"  Strike Price (K): ${config2.K}")
    print(f"  Option Type: {config2.option_type}")
    print(f"  Discrete Dividends:")
    for i, div in enumerate(config2.discrete_dividends):
        print(f"    {i+1}. ${div.amount} at t={div.time}")
    
    # Solve with discrete dividends
    S_grid2, t_grid2, V_grid2, div_indices2 = dividend_crank_nicolson_solver(
        config2, N_S=100, N_t=500
    )
    
    # Compare with analytical solution
    analytical_price2 = analytical_dividend_black_scholes(
        S_test, config2.K, config2.T, config2.r, config2.sigma, 
        config2.dividend_yield, config2.discrete_dividends, config2.option_type
    )
    
    S_idx2 = np.argmin(np.abs(S_grid2 - S_test))
    numerical_price2 = V_grid2[S_idx2, 0]
    
    print(f"\nResults at S=${S_test}, t=0:")
    print(f"  Analytical Price: ${analytical_price2:.6f}")
    print(f"  Numerical Price:  ${numerical_price2:.6f}")
    print(f"  Error: ${abs(analytical_price2 - numerical_price2):.6f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_dividend_results(S_grid1, t_grid1, V_grid1, config1, div_indices1)
    plot_dividend_results(S_grid2, t_grid2, V_grid2, config2, div_indices2)
    
    return (S_grid1, t_grid1, V_grid1, config1), (S_grid2, t_grid2, V_grid2, config2)


if __name__ == "__main__":
    # Run demonstration
    continuous_result, discrete_result = demonstrate_dividend_solver()
