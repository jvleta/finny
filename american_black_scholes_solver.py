"""
American Black-Scholes PDE Solver using Crank-Nicolson Method with Penalty Method

This module extends the European Black-Scholes solver to handle American options
by incorporating the early exercise constraint using the penalty method.

The American option pricing problem becomes a linear complementarity problem (LCP):
∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV ≤ 0
V(S,t) ≥ g(S,t)  (where g is the payoff function)
(∂V/∂t + LV)(V - g) = 0  (complementarity condition)

We solve this using the penalty method, which adds a penalty term when V < g.
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


class AmericanBlackScholesConfig:
    """Configuration class for American Black-Scholes parameters"""
    
    def __init__(self, 
                 S_max: float = 200.0,
                 K: float = 100.0,
                 T: float = 1.0,
                 r: float = 0.05,
                 sigma: float = 0.2,
                 option_type: str = 'put',  # American puts are more interesting
                 penalty_param: float = 1e6):
        """
        Initialize American Black-Scholes configuration
        
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
        penalty_param : float
            Penalty parameter for constraint enforcement
        """
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.penalty_param = penalty_param


def payoff_function(S: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """Calculate the payoff function for options"""
    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def boundary_conditions_american(S: np.ndarray, t: float, config: AmericanBlackScholesConfig) -> Tuple[float, float]:
    """
    Calculate boundary conditions for American options
    
    For American puts:
    - V(0,t) = K (immediate exercise)
    - V(S_max,t) = 0 (out of the money)
    
    For American calls:
    - V(0,t) = 0 (out of the money)  
    - V(S_max,t) = S_max - K*exp(-r*(T-t)) (deep in the money, same as European)
    """
    time_to_exp = config.T - t
    
    if config.option_type == 'call':
        lower_bc = 0.0
        upper_bc = max(0, config.S_max - config.K * np.exp(-config.r * time_to_exp))
    else:  # put
        lower_bc = config.K  # Early exercise is optimal at S=0
        upper_bc = 0.0
        
    return lower_bc, upper_bc


def penalty_method_step(V_old: np.ndarray, A, B, 
                       rhs: np.ndarray, payoff: np.ndarray, 
                       penalty_param: float) -> np.ndarray:
    """
    Solve one time step using the penalty method for American options
    
    The penalty method modifies the PDE to:
    ∂V/∂t + LV - ρ*max(g-V, 0) = 0
    
    where ρ is the penalty parameter and g is the payoff.
    """
    max_iterations = 50
    tolerance = 1e-8
    
    # Initial guess
    V_new = np.array(spsolve(A, rhs))
    
    for iteration in range(max_iterations):
        V_prev = V_new.copy()
        
        # Calculate penalty term
        penalty = penalty_param * np.maximum(payoff - V_new, 0)
        
        # Modified right-hand side with penalty
        rhs_penalty = rhs + penalty
        
        # Solve the modified system
        V_new = np.array(spsolve(A, rhs_penalty))
        
        # Check convergence
        if np.max(np.abs(V_new - V_prev)) < tolerance:
            break
    
    # Ensure the constraint V >= g is satisfied
    V_new = np.maximum(V_new, payoff)
    
    return V_new


def projected_sor_step(V_old: np.ndarray, A, B,
                      rhs: np.ndarray, payoff: np.ndarray, 
                      omega: float = 1.2) -> np.ndarray:
    """
    Projected SOR (Successive Over-Relaxation) method for American options
    
    This is an alternative to the penalty method that directly enforces
    the constraint V >= g at each iteration.
    """
    max_iterations = 100
    tolerance = 1e-8
    
    V_new = np.array(spsolve(A, rhs))  # Initial guess
    n = len(V_new)
    
    # Extract diagonal and off-diagonal parts of A
    A_diag = A.diagonal()
    
    for iteration in range(max_iterations):
        V_prev = V_new.copy()
        
        for i in range(n):
            # Calculate the unconstrained update
            residual = rhs[i] - A[i, :].dot(V_new)
            residual += A[i, i] * V_new[i]  # Add back diagonal term
            
            V_unconstrained = residual / A_diag[i]
            
            # Apply relaxation
            V_relaxed = (1 - omega) * V_new[i] + omega * V_unconstrained
            
            # Project onto constraint
            V_new[i] = max(V_relaxed, payoff[i])
        
        # Check convergence
        if np.max(np.abs(V_new - V_prev)) < tolerance:
            break
    
    return V_new


def american_crank_nicolson_solver(config: AmericanBlackScholesConfig, 
                                 N_S: int = 100, 
                                 N_t: int = 1000,
                                 method: str = 'penalty') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the American Black-Scholes PDE using Crank-Nicolson with constraint handling
    
    Parameters:
    -----------
    config : AmericanBlackScholesConfig
        Configuration object containing all parameters
    N_S : int
        Number of space grid points
    N_t : int
        Number of time grid points
    method : str
        'penalty' or 'projected_sor' for constraint handling
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        S_grid, t_grid, V_grid, exercise_boundary
    """
    
    # Setup grids
    S_grid = np.linspace(0, config.S_max, N_S)
    t_grid = np.linspace(0, config.T, N_t)
    
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]
    
    # Initialize solution matrix
    V = np.zeros((N_S, N_t))
    exercise_boundary = np.zeros(N_t)  # Track the exercise boundary
    
    # Set initial condition (payoff at expiry)
    V[:, -1] = payoff_function(S_grid, config.K, config.option_type)
    
    # Find initial exercise boundary
    payoff_at_expiry = V[:, -1]
    exercise_indices = np.where(payoff_at_expiry > 0)[0]
    if len(exercise_indices) > 0:
        exercise_boundary[-1] = S_grid[exercise_indices[-1]] if config.option_type == 'put' else S_grid[exercise_indices[0]]
    
    # Interior grid points
    interior_indices = np.arange(1, N_S - 1)
    S_interior = S_grid[interior_indices]
    
    # Coefficients for the finite difference scheme
    alpha = 0.5 * config.sigma**2 * S_interior**2 / dS**2
    beta = 0.5 * config.r * S_interior / dS
    gamma = config.r
    
    # Build matrices for Crank-Nicolson scheme
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
    
    print(f"Solving American {config.option_type} option using {method} method...")
    
    # Time stepping (backward in time)
    for j in range(N_t - 2, -1, -1):
        if j % (N_t // 10) == 0:
            progress = (N_t - 1 - j) / (N_t - 1) * 100
            print(f"Progress: {progress:.1f}% (time = {t_grid[j]:.3f})")
        
        # Get boundary conditions
        lower_bc, upper_bc = boundary_conditions_american(S_grid, t_grid[j], config)
        V[0, j] = lower_bc
        V[-1, j] = upper_bc
        
        # Right hand side
        rhs = B @ V[1:-1, j+1]
        
        # Add boundary condition contributions
        rhs[0] += 0.5 * dt * (alpha[0] - beta[0]) * (V[0, j+1] + V[0, j])
        rhs[-1] += 0.5 * dt * (alpha[-1] + beta[-1]) * (V[-1, j+1] + V[-1, j])
        
        # Payoff values for interior points
        payoff_interior = payoff_function(S_interior, config.K, config.option_type)
        
        # Solve with constraint handling
        if method == 'penalty':
            V[1:-1, j] = penalty_method_step(V[1:-1, j+1], A, B, rhs, 
                                           payoff_interior, config.penalty_param)
        elif method == 'projected_sor':
            V[1:-1, j] = projected_sor_step(V[1:-1, j+1], A, B, rhs, payoff_interior)
        else:
            raise ValueError("Method must be 'penalty' or 'projected_sor'")
        
        # Find exercise boundary (for puts, find the highest S where V = payoff)
        if config.option_type == 'put':
            payoff_all = payoff_function(S_grid, config.K, config.option_type)
            exercise_mask = np.abs(V[:, j] - payoff_all) < 1e-6
            exercise_indices = np.where(exercise_mask)[0]
            if len(exercise_indices) > 0:
                exercise_boundary[j] = S_grid[exercise_indices[-1]]
            else:
                exercise_boundary[j] = 0
        else:  # call
            payoff_all = payoff_function(S_grid, config.K, config.option_type)
            exercise_mask = np.abs(V[:, j] - payoff_all) < 1e-6
            exercise_indices = np.where(exercise_mask)[0]
            if len(exercise_indices) > 0:
                exercise_boundary[j] = S_grid[exercise_indices[0]]
            else:
                exercise_boundary[j] = config.S_max
    
    return S_grid, t_grid, V, exercise_boundary


def plot_american_results(S_grid: np.ndarray, t_grid: np.ndarray, V_grid: np.ndarray, 
                         exercise_boundary: np.ndarray, config: AmericanBlackScholesConfig):
    """
    Plot the results of the American option pricing
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
    ax1.set_title(f'American {config.option_type.title()} Option Value')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2. Contour plot with exercise boundary
    ax2 = axes[0, 1]
    contour = ax2.contourf(S_mesh, T_mesh, V_grid.T, levels=20, cmap='plasma')
    ax2.plot(exercise_boundary, t_grid, 'w-', linewidth=3, label='Exercise Boundary')
    ax2.set_xlabel('Stock Price (S)')
    ax2.set_ylabel('Time (t)')
    ax2.set_title('Option Value with Exercise Boundary')
    ax2.legend()
    plt.colorbar(contour, ax=ax2)
    
    # 3. Exercise boundary evolution
    ax3 = axes[0, 2]
    ax3.plot(t_grid, exercise_boundary, 'r-', linewidth=2)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Exercise Boundary (S*)')
    ax3.set_title('Exercise Boundary Evolution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Option value vs stock price at different times
    ax4 = axes[1, 0]
    time_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(time_levels)))
    
    for i, t_level in enumerate(time_levels):
        t_idx = np.argmin(np.abs(t_grid - t_level))
        ax4.plot(S_grid, V_grid[:, t_idx], color=colors[i], linewidth=2, 
                label=f't = {t_level:.2f}')
    
    # Add payoff
    payoff_values = payoff_function(S_grid, config.K, config.option_type)
    ax4.plot(S_grid, payoff_values, 'k--', linewidth=2, label='Payoff', alpha=0.7)
    
    ax4.set_xlabel('Stock Price (S)')
    ax4.set_ylabel('Option Value (V)')
    ax4.set_title('American Option Value vs Stock Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Early exercise premium
    ax5 = axes[1, 1]
    # Compare with European option (approximate)
    european_values = V_grid[:, 0].copy()  # This is actually American, but for illustration
    payoff_now = payoff_function(S_grid, config.K, config.option_type)
    exercise_premium = V_grid[:, 0] - payoff_now
    
    ax5.plot(S_grid, exercise_premium, 'b-', linewidth=2, label='Time Value')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Stock Price (S)')
    ax5.set_ylabel('Time Value (V - Payoff)')
    ax5.set_title('Time Value at t=0')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Comparison at specific stock prices
    ax6 = axes[1, 2]
    S_levels = [config.K * 0.8, config.K * 0.9, config.K, config.K * 1.1, config.K * 1.2]
    
    for S_level in S_levels:
        S_idx = np.argmin(np.abs(S_grid - S_level))
        moneyness = S_level / config.K
        label = f'S/K = {moneyness:.2f}'
        ax6.plot(t_grid, V_grid[S_idx, :], linewidth=2, label=label)
    
    ax6.set_xlabel('Time (t)')
    ax6.set_ylabel('Option Value (V)')
    ax6.set_title('Option Value Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_american_european(config: AmericanBlackScholesConfig, N_S: int = 100, N_t: int = 500):
    """
    Compare American and European option values
    """
    print("Comparing American vs European options...")
    
    # Solve American option
    S_grid, t_grid, V_american, exercise_boundary = american_crank_nicolson_solver(
        config, N_S, N_t, method='penalty'
    )
    
    # Import European solver
    from black_scholes_solver import BlackScholesConfig, crank_nicolson_solver
    
    # Solve European option with same parameters
    european_config = BlackScholesConfig(
        S_max=config.S_max, K=config.K, T=config.T,
        r=config.r, sigma=config.sigma, option_type=config.option_type
    )
    
    S_grid_eur, t_grid_eur, V_european = crank_nicolson_solver(european_config, N_S, N_t)
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # 1. Option values at t=0
    plt.subplot(2, 2, 1)
    plt.plot(S_grid, V_american[:, 0], 'r-', linewidth=2, label='American')
    plt.plot(S_grid_eur, V_european[:, 0], 'b--', linewidth=2, label='European')
    payoff_values = payoff_function(S_grid, config.K, config.option_type)
    plt.plot(S_grid, payoff_values, 'k:', linewidth=2, label='Payoff')
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Option Value ($)')
    plt.title('American vs European Option Values (t=0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Early exercise premium
    plt.subplot(2, 2, 2)
    premium = V_american[:, 0] - V_european[:, 0]
    plt.plot(S_grid, premium, 'g-', linewidth=2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Early Exercise Premium ($)')
    plt.title('Early Exercise Premium (American - European)')
    plt.grid(True, alpha=0.3)
    
    # 3. Exercise boundary
    plt.subplot(2, 2, 3)
    plt.plot(t_grid, exercise_boundary, 'r-', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Exercise Boundary (S*)')
    plt.title(f'Optimal Exercise Boundary ({config.option_type.title()})')
    plt.grid(True, alpha=0.3)
    
    # 4. Value evolution for ATM option
    plt.subplot(2, 2, 4)
    atm_idx = np.argmin(np.abs(S_grid - config.K))
    plt.plot(t_grid, V_american[atm_idx, :], 'r-', linewidth=2, label='American')
    plt.plot(t_grid_eur, V_european[atm_idx, :], 'b--', linewidth=2, label='European')
    plt.xlabel('Time (t)')
    plt.ylabel('Option Value ($)')
    plt.title('ATM Option Value Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    atm_american = V_american[atm_idx, 0]
    atm_european = V_european[atm_idx, 0]
    atm_premium = atm_american - atm_european
    
    print(f"\nAt-the-Money (S={config.K}) Comparison:")
    print(f"American Option Value: ${atm_american:.6f}")
    print(f"European Option Value: ${atm_european:.6f}")
    print(f"Early Exercise Premium: ${atm_premium:.6f}")
    print(f"Premium as % of European: {atm_premium/atm_european*100:.2f}%")
    
    return S_grid, t_grid, V_american, V_european, exercise_boundary


def demonstrate_american_solver():
    """
    Demonstrate the American option solver
    """
    print("American Black-Scholes PDE Solver")
    print("=" * 50)
    
    # Configuration for American put (most interesting case)
    config = AmericanBlackScholesConfig(
        S_max=150.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        option_type='put',
        penalty_param=1e6
    )
    
    print(f"Parameters:")
    print(f"  Strike Price (K): ${config.K}")
    print(f"  Time to Expiration (T): {config.T}")
    print(f"  Risk-free Rate (r): {config.r}")
    print(f"  Volatility (σ): {config.sigma}")
    print(f"  Option Type: {config.option_type}")
    print(f"  Penalty Parameter: {config.penalty_param}")
    
    # Solve American option
    S_grid, t_grid, V_grid, exercise_boundary = american_crank_nicolson_solver(
        config, N_S=100, N_t=500, method='penalty'
    )
    
    # Plot results
    plot_american_results(S_grid, t_grid, V_grid, exercise_boundary, config)
    
    # Compare with European
    compare_american_european(config)
    
    return S_grid, t_grid, V_grid, exercise_boundary


if __name__ == "__main__":
    # Run demonstration
    S_grid, t_grid, V_grid, exercise_boundary = demonstrate_american_solver()
