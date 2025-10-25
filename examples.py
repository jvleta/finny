#!/usr/bin/env python3
"""
Example script demonstrating the Black-Scholes PDE solver using Crank-Nicolson method

This script shows how to use the black_scholes_solver module to price options
and analyze their behavior.
"""

from black_scholes_solver import (
    BlackScholesConfig, 
    crank_nicolson_solver, 
    plot_results,
    analytical_black_scholes,
    demonstrate_solver
)
import numpy as np
import matplotlib.pyplot as plt


def example_1_basic_call_option():
    """Example 1: Basic call option pricing"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Call Option Pricing")
    print("=" * 60)
    
    # Configure parameters
    config = BlackScholesConfig(
        S_max=150.0,
        K=100.0,
        T=0.5,  # 6 months
        r=0.03,
        sigma=0.15,
        option_type='call'
    )
    
    # Solve
    S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=80, N_t=300)
    
    # Compare at-the-money option
    S_atm = config.K
    S_idx = np.argmin(np.abs(S_grid - S_atm))
    numerical_price = V_grid[S_idx, 0]
    analytical_price = analytical_black_scholes(S_atm, config.K, config.T, 
                                              config.r, config.sigma, 
                                              config.option_type)
    
    print(f"Configuration: K=${config.K}, T={config.T}yr, r={config.r*100}%, σ={config.sigma*100}%")
    print(f"At-the-money (S=${S_atm}) option prices:")
    print(f"  Numerical:  ${numerical_price:.6f}")
    print(f"  Analytical: ${analytical_price:.6f}")
    print(f"  Error:      ${abs(numerical_price - analytical_price):.6f}")
    
    return S_grid, t_grid, V_grid, config


def example_2_put_option_comparison():
    """Example 2: Put option with different volatilities"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Put Option with Different Volatilities")
    print("=" * 60)
    
    volatilities = [0.1, 0.2, 0.3, 0.4]
    results = {}
    
    plt.figure(figsize=(12, 8))
    
    for i, vol in enumerate(volatilities):
        config = BlackScholesConfig(
            S_max=200.0,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=vol,
            option_type='put'
        )
        
        S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=500)
        results[vol] = (S_grid, t_grid, V_grid)
        
        # Plot option value at t=0
        plt.subplot(2, 2, i+1)
        plt.plot(S_grid, V_grid[:, 0], 'b-', linewidth=2, label='Numerical')
        
        # Add analytical for comparison
        analytical_values = [analytical_black_scholes(s, config.K, config.T, 
                                                    config.r, config.sigma, 
                                                    config.option_type) 
                           for s in S_grid[1:-1]]
        plt.plot(S_grid[1:-1], analytical_values, 'r--', alpha=0.7, label='Analytical')
        
        plt.xlabel('Stock Price ($)')
        plt.ylabel('Put Option Value ($)')
        plt.title(f'Put Option (σ = {vol*100}%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Print at-the-money price
        atm_idx = np.argmin(np.abs(S_grid - config.K))
        atm_price = V_grid[atm_idx, 0]
        print(f"σ = {vol*100:2.0f}%: ATM Put Price = ${atm_price:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    return results


def example_3_time_to_expiration_effect():
    """Example 3: Effect of time to expiration"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Effect of Time to Expiration")
    print("=" * 60)
    
    time_periods = [0.25, 0.5, 1.0, 2.0]  # 3M, 6M, 1Y, 2Y
    
    plt.figure(figsize=(15, 10))
    
    for i, T in enumerate(time_periods):
        config = BlackScholesConfig(
            S_max=200.0,
            K=100.0,
            T=T,
            r=0.04,
            sigma=0.25,
            option_type='call'
        )
        
        S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=500)
        
        # Plot option value vs stock price at different times
        plt.subplot(2, 2, i+1)
        
        # Plot at different time points
        time_points = [0.0, T*0.25, T*0.5, T*0.75, T]
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(time_points)))
        
        for j, t_point in enumerate(time_points):
            t_idx = np.argmin(np.abs(t_grid - t_point))
            label = f't = {t_point:.2f}yr'
            plt.plot(S_grid, V_grid[:, t_idx], color=colors[j], 
                    linewidth=2, label=label)
        
        plt.xlabel('Stock Price ($)')
        plt.ylabel('Call Option Value ($)')
        plt.title(f'Call Option Evolution (T = {T}yr)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Print option value at different moneyness levels
        print(f"\nT = {T} years:")
        for S_level in [80, 90, 100, 110, 120]:
            S_idx = np.argmin(np.abs(S_grid - S_level))
            option_value = V_grid[S_idx, 0]
            moneyness = S_level / config.K
            print(f"  S=${S_level} (M={moneyness:.2f}): ${option_value:.4f}")
    
    plt.tight_layout()
    plt.show()


def example_4_greeks_analysis():
    """Example 4: Greeks analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Greeks Analysis")
    print("=" * 60)
    
    config = BlackScholesConfig(
        S_max=200.0,
        K=100.0,
        T=0.25,  # 3 months
        r=0.05,
        sigma=0.2,
        option_type='call'
    )
    
    S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=150, N_t=200)
    
    # Calculate numerical Greeks
    dS = S_grid[1] - S_grid[0]
    dt = t_grid[1] - t_grid[0]
    
    # Delta: ∂V/∂S
    delta = np.gradient(V_grid, dS, axis=0)
    
    # Gamma: ∂²V/∂S²
    gamma = np.gradient(delta, dS, axis=0)
    
    # Theta: -∂V/∂t
    theta = -np.gradient(V_grid, dt, axis=1)
    
    # Plot Greeks at t=0
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(S_grid, delta[:, 0], 'b-', linewidth=2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Delta')
    plt.title('Delta at t=0')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(S_grid, gamma[:, 0], 'g-', linewidth=2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Gamma')
    plt.title('Gamma at t=0')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(S_grid, theta[:, 0], 'r-', linewidth=2)
    plt.xlabel('Stock Price ($)')
    plt.ylabel('Theta')
    plt.title('Theta at t=0')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print Greeks at ATM
    atm_idx = np.argmin(np.abs(S_grid - config.K))
    print(f"Greeks at S=${config.K} (ATM), t=0:")
    print(f"  Delta: {delta[atm_idx, 0]:.6f}")
    print(f"  Gamma: {gamma[atm_idx, 0]:.6f}")
    print(f"  Theta: {theta[atm_idx, 0]:.6f}")


def example_5_parameter_sensitivity():
    """Example 5: Parameter sensitivity analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    base_config = BlackScholesConfig(K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call')
    
    # Test different parameters
    parameters = {
        'volatility': [0.1, 0.15, 0.2, 0.25, 0.3],
        'interest_rate': [0.01, 0.03, 0.05, 0.07, 0.09],
        'time_to_expiry': [0.25, 0.5, 1.0, 1.5, 2.0]
    }
    
    S_test = 100.0  # At-the-money
    
    print(f"Option prices at S=${S_test} (ATM):\n")
    
    # Volatility sensitivity
    print("Volatility Sensitivity:")
    for vol in parameters['volatility']:
        config = BlackScholesConfig(
            S_max=200.0, K=base_config.K, T=base_config.T, 
            r=base_config.r, sigma=vol, option_type=base_config.option_type
        )
        
        S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=300)
        S_idx = np.argmin(np.abs(S_grid - S_test))
        price = V_grid[S_idx, 0]
        
        print(f"  σ = {vol*100:2.0f}%: ${price:.4f}")
    
    # Interest rate sensitivity
    print("\nInterest Rate Sensitivity:")
    for rate in parameters['interest_rate']:
        config = BlackScholesConfig(
            S_max=200.0, K=base_config.K, T=base_config.T, 
            r=rate, sigma=base_config.sigma, option_type=base_config.option_type
        )
        
        S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=300)
        S_idx = np.argmin(np.abs(S_grid - S_test))
        price = V_grid[S_idx, 0]
        
        print(f"  r = {rate*100:2.0f}%: ${price:.4f}")
    
    # Time to expiry sensitivity
    print("\nTime to Expiry Sensitivity:")
    for time_exp in parameters['time_to_expiry']:
        config = BlackScholesConfig(
            S_max=200.0, K=base_config.K, T=time_exp, 
            r=base_config.r, sigma=base_config.sigma, option_type=base_config.option_type
        )
        
        S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=100, N_t=300)
        S_idx = np.argmin(np.abs(S_grid - S_test))
        price = V_grid[S_idx, 0]
        
        print(f"  T = {time_exp:.2f}yr: ${price:.4f}")


def main():
    """Run all examples"""
    print("Black-Scholes PDE Solver Examples")
    print("Using Crank-Nicolson Finite Difference Method")
    print("=" * 60)
    
    # Run all examples
    example_1_basic_call_option()
    example_2_put_option_comparison()
    example_3_time_to_expiration_effect()
    example_4_greeks_analysis()
    example_5_parameter_sensitivity()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
