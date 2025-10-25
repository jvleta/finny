"""
Unit tests for the Black-Scholes PDE solver

This module contains comprehensive unit tests for all functions and classes
in the black_scholes_solver.py module.
"""

import unittest
import numpy as np
import numpy.testing as npt
from scipy.stats import norm
import warnings

from black_scholes_solver import (
    BlackScholesConfig,
    payoff_function,
    boundary_conditions,
    crank_nicolson_solver,
    analytical_black_scholes,
    plot_results,
    demonstrate_solver
)


class TestBlackScholesConfig(unittest.TestCase):
    """Test the BlackScholesConfig class"""
    
    def test_default_initialization(self):
        """Test default parameter initialization"""
        config = BlackScholesConfig()
        
        self.assertEqual(config.S_max, 200.0)
        self.assertEqual(config.K, 100.0)
        self.assertEqual(config.T, 1.0)
        self.assertEqual(config.r, 0.05)
        self.assertEqual(config.sigma, 0.2)
        self.assertEqual(config.option_type, 'call')
    
    def test_custom_initialization(self):
        """Test custom parameter initialization"""
        config = BlackScholesConfig(
            S_max=150.0,
            K=110.0,
            T=0.5,
            r=0.03,
            sigma=0.25,
            option_type='put'
        )
        
        self.assertEqual(config.S_max, 150.0)
        self.assertEqual(config.K, 110.0)
        self.assertEqual(config.T, 0.5)
        self.assertEqual(config.r, 0.03)
        self.assertEqual(config.sigma, 0.25)
        self.assertEqual(config.option_type, 'put')
    
    def test_option_type_case_insensitive(self):
        """Test that option_type is case insensitive"""
        config1 = BlackScholesConfig(option_type='CALL')
        config2 = BlackScholesConfig(option_type='Put')
        
        self.assertEqual(config1.option_type, 'call')
        self.assertEqual(config2.option_type, 'put')


class TestPayoffFunction(unittest.TestCase):
    """Test the payoff_function"""
    
    def setUp(self):
        """Set up test data"""
        self.S = np.array([80, 90, 100, 110, 120])
        self.K = 100.0
    
    def test_call_payoff(self):
        """Test call option payoff"""
        expected = np.array([0, 0, 0, 10, 20])
        result = payoff_function(self.S, self.K, 'call')
        npt.assert_array_equal(result, expected)
    
    def test_put_payoff(self):
        """Test put option payoff"""
        expected = np.array([20, 10, 0, 0, 0])
        result = payoff_function(self.S, self.K, 'put')
        npt.assert_array_equal(result, expected)
    
    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError"""
        with self.assertRaises(ValueError):
            payoff_function(self.S, self.K, 'invalid')
    
    def test_single_value_input(self):
        """Test with single value input"""
        result_call = payoff_function(np.array([110]), self.K, 'call')
        result_put = payoff_function(np.array([90]), self.K, 'put')
        
        self.assertEqual(result_call[0], 10)
        self.assertEqual(result_put[0], 10)


class TestBoundaryConditions(unittest.TestCase):
    """Test the boundary_conditions function"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = BlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2
        )
        self.S = np.linspace(0, self.config.S_max, 100)
    
    def test_call_boundary_conditions(self):
        """Test boundary conditions for call option"""
        self.config.option_type = 'call'
        
        # At t=0 (start)
        lower_bc, upper_bc = boundary_conditions(self.S, 0.0, self.config)
        self.assertEqual(lower_bc, 0.0)
        expected_upper = self.config.S_max - self.config.K * np.exp(-self.config.r * self.config.T)
        self.assertAlmostEqual(upper_bc, expected_upper, places=6)
        
        # At t=T (expiry)
        lower_bc, upper_bc = boundary_conditions(self.S, self.config.T, self.config)
        self.assertEqual(lower_bc, 0.0)
        self.assertAlmostEqual(upper_bc, self.config.S_max - self.config.K, places=6)
    
    def test_put_boundary_conditions(self):
        """Test boundary conditions for put option"""
        self.config.option_type = 'put'
        
        # At t=0 (start)
        lower_bc, upper_bc = boundary_conditions(self.S, 0.0, self.config)
        expected_lower = self.config.K * np.exp(-self.config.r * self.config.T)
        self.assertAlmostEqual(lower_bc, expected_lower, places=6)
        self.assertEqual(upper_bc, 0.0)
        
        # At t=T (expiry)
        lower_bc, upper_bc = boundary_conditions(self.S, self.config.T, self.config)
        self.assertAlmostEqual(lower_bc, self.config.K, places=6)
        self.assertEqual(upper_bc, 0.0)


class TestAnalyticalBlackScholes(unittest.TestCase):
    """Test the analytical_black_scholes function"""
    
    def test_call_option_known_values(self):
        """Test call option with known analytical values"""
        # Parameters from a well-known example
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        
        result = analytical_black_scholes(S, K, T, r, sigma, 'call')
        
        # The exact value should be around 10.45 for these parameters
        self.assertAlmostEqual(result, 10.4506, places=3)
    
    def test_put_option_known_values(self):
        """Test put option with known analytical values"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        
        result = analytical_black_scholes(S, K, T, r, sigma, 'put')
        
        # Using put-call parity: Put = Call - S + K*exp(-r*T)
        call_value = analytical_black_scholes(S, K, T, r, sigma, 'call')
        expected_put = call_value - S + K * np.exp(-r * T)
        
        self.assertAlmostEqual(result, expected_put, places=6)
    
    def test_deep_itm_call(self):
        """Test deep in-the-money call option"""
        result = analytical_black_scholes(150, 100, 1, 0.05, 0.2, 'call')
        
        # Should be approximately S - K*exp(-r*T) for deep ITM
        expected_min = 150 - 100 * np.exp(-0.05 * 1)
        self.assertGreater(result, expected_min)
    
    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option"""
        result = analytical_black_scholes(50, 100, 1, 0.05, 0.2, 'call')
        
        # Should be small but positive
        self.assertGreater(result, 0)
        self.assertLess(result, 5)  # Should be less than 5 for these parameters
    
    def test_zero_volatility(self):
        """Test with zero volatility"""
        S, K, T, r = 110, 100, 1, 0.05
        
        # With zero volatility, call value should be max(S - K*exp(-r*T), 0)
        result = analytical_black_scholes(S, K, T, r, 0.0, 'call')
        expected = max(S - K * np.exp(-r * T), 0)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_zero_time_to_expiry(self):
        """Test with zero time to expiry"""
        S, K, r, sigma = 110, 100, 0.05, 0.2
        
        # With T=0, should equal payoff
        call_result = analytical_black_scholes(S, K, 0.0, r, sigma, 'call')
        put_result = analytical_black_scholes(S, K, 0.0, r, sigma, 'put')
        
        self.assertAlmostEqual(call_result, max(S - K, 0), places=6)
        self.assertAlmostEqual(put_result, max(K - S, 0), places=6)


class TestCrankNicolsonSolver(unittest.TestCase):
    """Test the crank_nicolson_solver function"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = BlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
    
    def test_solver_output_shapes(self):
        """Test that solver returns correct output shapes"""
        N_S, N_t = 50, 100
        S_grid, t_grid, V_grid = crank_nicolson_solver(self.config, N_S, N_t)
        
        self.assertEqual(len(S_grid), N_S)
        self.assertEqual(len(t_grid), N_t)
        self.assertEqual(V_grid.shape, (N_S, N_t))
    
    def test_initial_condition(self):
        """Test that initial condition (payoff) is correctly set"""
        N_S, N_t = 50, 100
        S_grid, t_grid, V_grid = crank_nicolson_solver(self.config, N_S, N_t)
        
        # Final time values should equal payoff
        expected_payoff = payoff_function(S_grid, self.config.K, self.config.option_type)
        npt.assert_array_almost_equal(V_grid[:, -1], expected_payoff, decimal=10)
    
    def test_boundary_conditions_consistency(self):
        """Test that boundary conditions are maintained"""
        N_S, N_t = 50, 100
        S_grid, t_grid, V_grid = crank_nicolson_solver(self.config, N_S, N_t)
        
        for j in range(N_t):
            lower_bc, upper_bc = boundary_conditions(S_grid, t_grid[j], self.config)
            self.assertAlmostEqual(V_grid[0, j], lower_bc, places=6)
            self.assertAlmostEqual(V_grid[-1, j], upper_bc, places=6)
    
    def test_monotonicity_call(self):
        """Test monotonicity properties for call option"""
        self.config.option_type = 'call'
        N_S, N_t = 50, 100
        S_grid, t_grid, V_grid = crank_nicolson_solver(self.config, N_S, N_t)
        
        # For call options, value should be non-decreasing in S
        for j in range(N_t):
            differences = np.diff(V_grid[:, j])
            self.assertTrue(np.all(differences >= -1e-6))  # Allow small numerical errors
    
    def test_put_call_parity_approximation(self):
        """Test approximate put-call parity relationship"""
        # Solve for call
        call_config = BlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        S_grid_c, t_grid_c, V_grid_c = crank_nicolson_solver(call_config, 50, 100)
        
        # Solve for put with same parameters
        put_config = BlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='put'
        )
        S_grid_p, t_grid_p, V_grid_p = crank_nicolson_solver(put_config, 50, 100)
        
        # Check put-call parity at t=0: C - P = S - K*exp(-r*T)
        S_test_idx = 25  # Middle of the grid
        S_test = S_grid_c[S_test_idx]
        
        call_value = V_grid_c[S_test_idx, 0]
        put_value = V_grid_p[S_test_idx, 0]
        
        parity_lhs = call_value - put_value
        parity_rhs = S_test - call_config.K * np.exp(-call_config.r * call_config.T)
        
        # Allow for some numerical error
        self.assertAlmostEqual(parity_lhs, parity_rhs, delta=0.5)
    
    def test_convergence_to_analytical(self):
        """Test convergence to analytical solution"""
        # Test with moderate grid sizes
        N_S, N_t = 100, 500
        S_grid, t_grid, V_grid = crank_nicolson_solver(self.config, N_S, N_t)
        
        # Compare at several points
        test_prices = [80, 100, 120]
        
        for S_test in test_prices:
            S_idx = np.argmin(np.abs(S_grid - S_test))
            numerical_value = V_grid[S_idx, 0]
            
            analytical_value = analytical_black_scholes(
                S_test, self.config.K, self.config.T, 
                self.config.r, self.config.sigma, self.config.option_type
            )
            
            relative_error = abs(numerical_value - analytical_value) / analytical_value
            self.assertLess(relative_error, 0.15)  # Less than 15% error (finite difference approximation)


class TestPlottingFunctions(unittest.TestCase):
    """Test plotting functions (mainly for execution without errors)"""
    
    def setUp(self):
        """Set up test data"""
        self.config = BlackScholesConfig()
        # Use small grids for faster testing
        self.S_grid, self.t_grid, self.V_grid = crank_nicolson_solver(
            self.config, N_S=20, N_t=50
        )
    
    def test_plot_results_execution(self):
        """Test that plot_results executes without error"""
        # Suppress matplotlib warnings during testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                # Import matplotlib and set backend to prevent display issues in tests
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                
                plot_results(self.S_grid, self.t_grid, self.V_grid, self.config)
                
                # If we get here, the function executed successfully
                self.assertTrue(True)
                
            except Exception as e:
                self.fail(f"plot_results raised an exception: {e}")
    
    def test_demonstrate_solver_execution(self):
        """Test that demonstrate_solver executes without error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                
                # This will run the full demonstration
                # We'll catch any exceptions
                result = demonstrate_solver()
                
                # Check that it returns the expected tuple
                self.assertEqual(len(result), 4)
                S_grid, t_grid, V_grid, config = result
                
                self.assertIsInstance(S_grid, np.ndarray)
                self.assertIsInstance(t_grid, np.ndarray)
                self.assertIsInstance(V_grid, np.ndarray)
                self.assertIsInstance(config, BlackScholesConfig)
                
            except Exception as e:
                self.fail(f"demonstrate_solver raised an exception: {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_very_small_volatility(self):
        """Test with very small volatility"""
        config = BlackScholesConfig(sigma=1e-6)
        
        try:
            S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=20, N_t=50)
            
            # Should not raise an exception
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Small volatility case failed: {e}")
    
    def test_very_small_time_to_expiry(self):
        """Test with very small time to expiry"""
        config = BlackScholesConfig(T=1e-3)
        
        try:
            S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=20, N_t=10)
            
            # Should not raise an exception
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Small time to expiry case failed: {e}")
    
    def test_zero_interest_rate(self):
        """Test with zero interest rate"""
        config = BlackScholesConfig(r=0.0)
        
        try:
            S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=20, N_t=50)
            
            # Should not raise an exception
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Zero interest rate case failed: {e}")
    
    def test_high_volatility(self):
        """Test with high volatility"""
        config = BlackScholesConfig(sigma=1.0)  # 100% volatility
        
        try:
            S_grid, t_grid, V_grid = crank_nicolson_solver(config, N_S=50, N_t=100)
            
            # Values should still be reasonable (no negative values for calls)
            if config.option_type == 'call':
                self.assertTrue(np.all(V_grid >= -1e-10))  # Allow tiny numerical errors
            
        except Exception as e:
            self.fail(f"High volatility case failed: {e}")


if __name__ == '__main__':
    # Set matplotlib backend for testing
    import matplotlib
    matplotlib.use('Agg')
    
    # Run all tests
    unittest.main(verbosity=2)
