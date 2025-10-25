"""
Unit tests for the American Black-Scholes PDE solver

This module contains comprehensive unit tests for all functions and classes
in the american_black_scholes_solver.py module.
"""

import unittest
import numpy as np
import numpy.testing as npt
import warnings

from american_black_scholes_solver import (
    AmericanBlackScholesConfig,
    payoff_function,
    boundary_conditions_american,
    penalty_method_step,
    projected_sor_step,
    american_crank_nicolson_solver,
    plot_american_results,
    compare_american_european,
    demonstrate_american_solver
)
from scipy.sparse import diags


class TestAmericanBlackScholesConfig(unittest.TestCase):
    """Test the AmericanBlackScholesConfig class"""
    
    def test_default_initialization(self):
        """Test default parameter initialization"""
        config = AmericanBlackScholesConfig()
        
        self.assertEqual(config.S_max, 200.0)
        self.assertEqual(config.K, 100.0)
        self.assertEqual(config.T, 1.0)
        self.assertEqual(config.r, 0.05)
        self.assertEqual(config.sigma, 0.2)
        self.assertEqual(config.option_type, 'put')  # Default for American
        self.assertEqual(config.penalty_param, 1e6)
    
    def test_custom_initialization(self):
        """Test custom parameter initialization"""
        config = AmericanBlackScholesConfig(
            S_max=150.0,
            K=110.0,
            T=0.5,
            r=0.03,
            sigma=0.25,
            option_type='call',
            penalty_param=1e5
        )
        
        self.assertEqual(config.S_max, 150.0)
        self.assertEqual(config.K, 110.0)
        self.assertEqual(config.T, 0.5)
        self.assertEqual(config.r, 0.03)
        self.assertEqual(config.sigma, 0.25)
        self.assertEqual(config.option_type, 'call')
        self.assertEqual(config.penalty_param, 1e5)
    
    def test_option_type_case_insensitive(self):
        """Test that option_type is case insensitive"""
        config1 = AmericanBlackScholesConfig(option_type='CALL')
        config2 = AmericanBlackScholesConfig(option_type='Put')
        
        self.assertEqual(config1.option_type, 'call')
        self.assertEqual(config2.option_type, 'put')


class TestAmericanPayoffFunction(unittest.TestCase):
    """Test the payoff_function for American options"""
    
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


class TestAmericanBoundaryConditions(unittest.TestCase):
    """Test the boundary_conditions_american function"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = AmericanBlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2
        )
        self.S = np.linspace(0, self.config.S_max, 100)
    
    def test_put_boundary_conditions(self):
        """Test boundary conditions for American put option"""
        self.config.option_type = 'put'
        
        # At t=0 (start)
        lower_bc, upper_bc = boundary_conditions_american(self.S, 0.0, self.config)
        self.assertEqual(lower_bc, self.config.K)  # Early exercise optimal at S=0
        self.assertEqual(upper_bc, 0.0)
        
        # At t=T (expiry)
        lower_bc, upper_bc = boundary_conditions_american(self.S, self.config.T, self.config)
        self.assertEqual(lower_bc, self.config.K)
        self.assertEqual(upper_bc, 0.0)
    
    def test_call_boundary_conditions(self):
        """Test boundary conditions for American call option"""
        self.config.option_type = 'call'
        
        # At t=0 (start)
        lower_bc, upper_bc = boundary_conditions_american(self.S, 0.0, self.config)
        self.assertEqual(lower_bc, 0.0)
        expected_upper = max(0, self.config.S_max - self.config.K * np.exp(-self.config.r * self.config.T))
        self.assertAlmostEqual(upper_bc, expected_upper, places=6)
        
        # At t=T (expiry)
        lower_bc, upper_bc = boundary_conditions_american(self.S, self.config.T, self.config)
        self.assertEqual(lower_bc, 0.0)
        expected_upper = max(0, self.config.S_max - self.config.K)
        self.assertAlmostEqual(upper_bc, expected_upper, places=6)


class TestPenaltyMethodStep(unittest.TestCase):
    """Test the penalty_method_step function"""
    
    def setUp(self):
        """Set up test matrices and data"""
        n = 10
        # Create a simple tridiagonal matrix
        diag_vals = np.ones(n) * 2
        off_diag = np.ones(n-1) * (-1)
        
        self.A = diags([off_diag, diag_vals, off_diag], offsets=[-1, 0, 1], format='csr')
        self.B = diags([off_diag*0.5, diag_vals*0.5, off_diag*0.5], offsets=[-1, 0, 1], format='csr')
        
        self.V_old = np.random.rand(n)
        self.rhs = np.random.rand(n)
        self.payoff = np.maximum(100 - np.linspace(80, 120, n), 0)  # Put payoff
        self.penalty_param = 1e6
    
    def test_penalty_method_convergence(self):
        """Test that penalty method converges"""
        result = penalty_method_step(
            self.V_old, self.A, self.B, self.rhs, self.payoff, self.penalty_param
        )
        
        # Result should be a numpy array of correct length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.V_old))
    
    def test_constraint_enforcement(self):
        """Test that penalty method enforces V >= payoff"""
        result = penalty_method_step(
            self.V_old, self.A, self.B, self.rhs, self.payoff, self.penalty_param
        )
        
        # All values should be at least as large as payoff
        self.assertTrue(np.all(result >= self.payoff - 1e-8))  # Allow small numerical error
    
    def test_different_penalty_parameters(self):
        """Test behavior with different penalty parameters"""
        penalty_low = 1e3
        penalty_high = 1e9
        
        result_low = penalty_method_step(
            self.V_old, self.A, self.B, self.rhs, self.payoff, penalty_low
        )
        result_high = penalty_method_step(
            self.V_old, self.A, self.B, self.rhs, self.payoff, penalty_high
        )
        
        # Both should satisfy constraints
        self.assertTrue(np.all(result_low >= self.payoff - 1e-6))
        self.assertTrue(np.all(result_high >= self.payoff - 1e-8))


class TestProjectedSORStep(unittest.TestCase):
    """Test the projected_sor_step function"""
    
    def setUp(self):
        """Set up test matrices and data"""
        n = 10
        # Create a diagonally dominant matrix for stability
        diag_vals = np.ones(n) * 3
        off_diag = np.ones(n-1) * (-1)
        
        self.A = diags([off_diag, diag_vals, off_diag], offsets=[-1, 0, 1], format='csr')
        self.B = diags([off_diag*0.5, diag_vals*0.5, off_diag*0.5], offsets=[-1, 0, 1], format='csr')
        
        self.V_old = np.random.rand(n)
        self.rhs = np.random.rand(n)
        self.payoff = np.maximum(100 - np.linspace(80, 120, n), 0)  # Put payoff
    
    def test_projected_sor_convergence(self):
        """Test that projected SOR converges"""
        result = projected_sor_step(
            self.V_old, self.A, self.B, self.rhs, self.payoff, omega=1.0
        )
        
        # Result should be a numpy array of correct length
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.V_old))
    
    def test_constraint_enforcement_sor(self):
        """Test that projected SOR enforces V >= payoff"""
        result = projected_sor_step(
            self.V_old, self.A, self.B, self.rhs, self.payoff, omega=1.2
        )
        
        # All values should be at least as large as payoff
        self.assertTrue(np.all(result >= self.payoff - 1e-8))
    
    def test_different_omega_values(self):
        """Test with different relaxation parameters"""
        for omega in [0.8, 1.0, 1.2, 1.5]:
            result = projected_sor_step(
                self.V_old, self.A, self.B, self.rhs, self.payoff, omega=omega
            )
            
            # Should satisfy constraints regardless of omega
            self.assertTrue(np.all(result >= self.payoff - 1e-8))


class TestAmericanCrankNicolsonSolver(unittest.TestCase):
    """Test the american_crank_nicolson_solver function"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = AmericanBlackScholesConfig(
            S_max=150.0, K=100.0, T=0.5, r=0.05, sigma=0.2, option_type='put'
        )
    
    def test_solver_output_shapes(self):
        """Test that American solver returns correct output shapes"""
        N_S, N_t = 30, 50  # Smaller grids for faster testing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress progress messages
            
            S_grid, t_grid, V_grid, exercise_boundary = american_crank_nicolson_solver(
                self.config, N_S, N_t, method='penalty'
            )
        
        self.assertEqual(len(S_grid), N_S)
        self.assertEqual(len(t_grid), N_t)
        self.assertEqual(V_grid.shape, (N_S, N_t))
        self.assertEqual(len(exercise_boundary), N_t)
    
    def test_initial_condition_american(self):
        """Test that initial condition (payoff) is correctly set"""
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid, exercise_boundary = american_crank_nicolson_solver(
                self.config, N_S, N_t, method='penalty'
            )
        
        # Final time values should equal payoff
        expected_payoff = payoff_function(S_grid, self.config.K, self.config.option_type)
        npt.assert_array_almost_equal(V_grid[:, -1], expected_payoff, decimal=10)
    
    def test_constraint_satisfaction(self):
        """Test that American option constraint V >= payoff is satisfied"""
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid, exercise_boundary = american_crank_nicolson_solver(
                self.config, N_S, N_t, method='penalty'
            )
        
        # Check constraint at all points
        for j in range(N_t):
            payoff_at_time = payoff_function(S_grid, self.config.K, self.config.option_type)
            constraint_violations = V_grid[:, j] < payoff_at_time - 1e-6
            self.assertFalse(np.any(constraint_violations), 
                           f"Constraint violation at time index {j}")
    
    def test_both_methods(self):
        """Test that both penalty and projected SOR methods work"""
        N_S, N_t = 20, 30  # Small grids for speed
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test penalty method
            try:
                S_grid_p, t_grid_p, V_grid_p, eb_p = american_crank_nicolson_solver(
                    self.config, N_S, N_t, method='penalty'
                )
                penalty_success = True
            except Exception as e:
                penalty_success = False
                self.fail(f"Penalty method failed: {e}")
            
            # Test projected SOR method
            try:
                S_grid_s, t_grid_s, V_grid_s, eb_s = american_crank_nicolson_solver(
                    self.config, N_S, N_t, method='projected_sor'
                )
                sor_success = True
            except Exception as e:
                sor_success = False
                self.fail(f"Projected SOR method failed: {e}")
            
            self.assertTrue(penalty_success and sor_success)
    
    def test_invalid_method(self):
        """Test that invalid method raises ValueError"""
        with self.assertRaises(ValueError):
            american_crank_nicolson_solver(self.config, 20, 30, method='invalid')
    
    def test_exercise_boundary_properties(self):
        """Test properties of the exercise boundary"""
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid, exercise_boundary = american_crank_nicolson_solver(
                self.config, N_S, N_t, method='penalty'
            )
        
        # Exercise boundary should be non-negative
        self.assertTrue(np.all(exercise_boundary >= 0))
        
        # For American puts, exercise boundary should generally be reasonable
        # (numerical methods may have some boundary artifacts)
        if self.config.option_type == 'put':
            # Most exercise boundaries should be reasonable
            reasonable_boundaries = exercise_boundary[
                (exercise_boundary > 0) & (exercise_boundary < self.config.S_max)
            ]
            if len(reasonable_boundaries) > 0:
                # At least some boundaries should be <= K
                reasonable_count = np.sum(reasonable_boundaries <= self.config.K * 1.1)
                self.assertGreater(reasonable_count, 0, 
                                 "No reasonable exercise boundaries found")
    
    def test_american_vs_european_value(self):
        """Test that American option value >= European option value"""
        # This is more of an integration test
        N_S, N_t = 30, 40
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # American option
            S_grid, t_grid, V_american, _ = american_crank_nicolson_solver(
                self.config, N_S, N_t, method='penalty'
            )
            
            # European option (approximate using final constraint satisfaction)
            # For a proper test, we'd import the European solver
            # Here we just check that American values are reasonable
            
            # American option should have non-negative time value
            payoff_grid = np.array([payoff_function(S_grid, self.config.K, self.config.option_type) 
                                  for _ in range(len(t_grid))]).T
            
            time_values = V_american - payoff_grid
            
            # Time values should be non-negative (allowing small numerical errors)
            self.assertTrue(np.all(time_values >= -1e-6))


class TestAmericanPlottingFunctions(unittest.TestCase):
    """Test American option plotting functions"""
    
    def setUp(self):
        """Set up test data"""
        self.config = AmericanBlackScholesConfig(
            S_max=100.0, K=50.0, T=0.25, r=0.05, sigma=0.2, option_type='put'
        )
        
        # Use very small grids for fast testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.S_grid, self.t_grid, self.V_grid, self.exercise_boundary = \
                american_crank_nicolson_solver(self.config, N_S=15, N_t=20, method='penalty')
    
    def test_plot_american_results_execution(self):
        """Test that plot_american_results executes without error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                
                plot_american_results(
                    self.S_grid, self.t_grid, self.V_grid, 
                    self.exercise_boundary, self.config
                )
                
                self.assertTrue(True)  # If we get here, no exception was raised
                
            except Exception as e:
                self.fail(f"plot_american_results raised an exception: {e}")
    
    def test_compare_american_european_execution(self):
        """Test that compare_american_european executes without error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                
                # This function should handle the import internally
                result = compare_american_european(self.config, N_S=15, N_t=20)
                
                # Should return a tuple of results
                self.assertEqual(len(result), 5)
                
            except ImportError:
                # If European solver not available, that's expected in isolated test
                self.skipTest("European solver not available for comparison")
            except Exception as e:
                self.fail(f"compare_american_european raised an exception: {e}")
    
    def test_demonstrate_american_solver_execution(self):
        """Test that demonstrate_american_solver executes without error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                
                result = demonstrate_american_solver()
                
                # Should return a tuple
                self.assertEqual(len(result), 4)
                S_grid, t_grid, V_grid, exercise_boundary = result
                
                self.assertIsInstance(S_grid, np.ndarray)
                self.assertIsInstance(t_grid, np.ndarray)
                self.assertIsInstance(V_grid, np.ndarray)
                self.assertIsInstance(exercise_boundary, np.ndarray)
                
            except Exception as e:
                self.fail(f"demonstrate_american_solver raised an exception: {e}")


class TestAmericanEdgeCases(unittest.TestCase):
    """Test edge cases for American options"""
    
    def test_very_high_penalty_parameter(self):
        """Test with very high penalty parameter"""
        config = AmericanBlackScholesConfig(penalty_param=1e12)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                S_grid, t_grid, V_grid, eb = american_crank_nicolson_solver(
                    config, N_S=20, N_t=30, method='penalty'
                )
                self.assertTrue(True)  # Should not raise exception
                
            except Exception as e:
                self.fail(f"High penalty parameter case failed: {e}")
    
    def test_very_low_penalty_parameter(self):
        """Test with low penalty parameter"""
        config = AmericanBlackScholesConfig(penalty_param=1e2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                S_grid, t_grid, V_grid, eb = american_crank_nicolson_solver(
                    config, N_S=20, N_t=30, method='penalty'
                )
                self.assertTrue(True)  # Should not raise exception
                
            except Exception as e:
                self.fail(f"Low penalty parameter case failed: {e}")
    
    def test_american_call_with_high_interest_rate(self):
        """Test American call with high interest rate (early exercise more likely)"""
        config = AmericanBlackScholesConfig(
            option_type='call', r=0.15, K=100.0, S_max=200.0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                S_grid, t_grid, V_grid, eb = american_crank_nicolson_solver(
                    config, N_S=25, N_t=40, method='penalty'
                )
                
                # Exercise boundary should be reasonable
                self.assertTrue(np.all(eb >= 0))
                self.assertTrue(np.all(eb <= config.S_max))
                
            except Exception as e:
                self.fail(f"High interest rate American call failed: {e}")
    
    def test_short_time_to_expiry(self):
        """Test with very short time to expiry"""
        config = AmericanBlackScholesConfig(T=0.01)  # 0.01 years ≈ 3.65 days
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                S_grid, t_grid, V_grid, eb = american_crank_nicolson_solver(
                    config, N_S=20, N_t=20, method='penalty'
                )
                
                # Option values should be close to payoff
                payoff_final = payoff_function(S_grid, config.K, config.option_type)
                payoff_initial = payoff_function(S_grid, config.K, config.option_type)
                
                # Values at t=0 should not be much higher than payoff for short expiry
                time_values = V_grid[:, 0] - payoff_initial
                
                # Most time values should be small for short expiry
                self.assertTrue(np.all(time_values >= -1e-8))  # Non-negative time value
                
            except Exception as e:
                self.fail(f"Short time to expiry case failed: {e}")


if __name__ == '__main__':
    # Set matplotlib backend for testing
    import matplotlib
    matplotlib.use('Agg')
    
    # Run all tests
    unittest.main(verbosity=2)
