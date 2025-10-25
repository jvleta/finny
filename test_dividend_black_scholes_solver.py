"""
Unit tests for the Dividend-Enhanced Black-Scholes PDE solver

This module contains comprehensive unit tests for all functions and classes
in the dividend_black_scholes_solver.py module.
"""

import unittest
import numpy as np
import numpy.testing as npt
import warnings

from dividend_black_scholes_solver import (
    DividendEvent,
    DividendBlackScholesConfig,
    payoff_function_dividend,
    boundary_conditions_dividend,
    adjust_stock_price_for_dividends,
    interpolate_option_values,
    dividend_crank_nicolson_solver,
    analytical_dividend_black_scholes,
    plot_dividend_results,
    demonstrate_dividend_solver
)


class TestDividendEvent(unittest.TestCase):
    """Test the DividendEvent dataclass"""
    
    def test_valid_dividend_event(self):
        """Test creation of valid dividend event"""
        div = DividendEvent(time=0.5, amount=1.0)
        self.assertEqual(div.time, 0.5)
        self.assertEqual(div.amount, 1.0)
    
    def test_negative_time_raises_error(self):
        """Test that negative time raises ValueError"""
        with self.assertRaises(ValueError):
            DividendEvent(time=-0.1, amount=1.0)
    
    def test_negative_amount_raises_error(self):
        """Test that negative amount raises ValueError"""
        with self.assertRaises(ValueError):
            DividendEvent(time=0.5, amount=-1.0)
    
    def test_zero_values_allowed(self):
        """Test that zero values are allowed"""
        div1 = DividendEvent(time=0.0, amount=1.0)
        div2 = DividendEvent(time=0.5, amount=0.0)
        self.assertEqual(div1.time, 0.0)
        self.assertEqual(div2.amount, 0.0)


class TestDividendBlackScholesConfig(unittest.TestCase):
    """Test the DividendBlackScholesConfig class"""
    
    def test_default_initialization(self):
        """Test default parameter initialization"""
        config = DividendBlackScholesConfig()
        
        self.assertEqual(config.S_max, 200.0)
        self.assertEqual(config.K, 100.0)
        self.assertEqual(config.T, 1.0)
        self.assertEqual(config.r, 0.05)
        self.assertEqual(config.sigma, 0.2)
        self.assertEqual(config.option_type, 'call')
        self.assertEqual(config.dividend_yield, 0.0)
        self.assertEqual(len(config.discrete_dividends), 0)
    
    def test_continuous_dividend_initialization(self):
        """Test initialization with continuous dividend yield"""
        config = DividendBlackScholesConfig(
            dividend_yield=0.03,
            option_type='put'
        )
        
        self.assertEqual(config.dividend_yield, 0.03)
        self.assertEqual(config.option_type, 'put')
    
    def test_discrete_dividend_initialization(self):
        """Test initialization with discrete dividends"""
        dividends = [
            DividendEvent(time=0.25, amount=1.0),
            DividendEvent(time=0.75, amount=1.5)
        ]
        
        config = DividendBlackScholesConfig(
            T=1.0,
            discrete_dividends=dividends
        )
        
        self.assertEqual(len(config.discrete_dividends), 2)
        self.assertEqual(config.discrete_dividends[0].amount, 1.0)
        self.assertEqual(config.discrete_dividends[1].amount, 1.5)
    
    def test_dividend_sorting(self):
        """Test that dividends are sorted by time"""
        dividends = [
            DividendEvent(time=0.75, amount=1.5),
            DividendEvent(time=0.25, amount=1.0),
            DividendEvent(time=0.50, amount=1.2)
        ]
        
        config = DividendBlackScholesConfig(
            T=1.0,
            discrete_dividends=dividends
        )
        
        # Should be sorted by time
        times = [div.time for div in config.discrete_dividends]
        self.assertEqual(times, [0.25, 0.50, 0.75])
    
    def test_dividend_after_expiry_raises_error(self):
        """Test that dividend after expiry raises ValueError"""
        dividends = [
            DividendEvent(time=1.5, amount=1.0)  # After T=1.0
        ]
        
        with self.assertRaises(ValueError):
            DividendBlackScholesConfig(
                T=1.0,
                discrete_dividends=dividends
            )
    
    def test_option_type_case_insensitive(self):
        """Test that option_type is case insensitive"""
        config1 = DividendBlackScholesConfig(option_type='CALL')
        config2 = DividendBlackScholesConfig(option_type='Put')
        
        self.assertEqual(config1.option_type, 'call')
        self.assertEqual(config2.option_type, 'put')


class TestPayoffFunctionDividend(unittest.TestCase):
    """Test the payoff_function_dividend"""
    
    def setUp(self):
        """Set up test data"""
        self.S = np.array([80, 90, 100, 110, 120])
        self.K = 100.0
    
    def test_call_payoff(self):
        """Test call option payoff"""
        expected = np.array([0, 0, 0, 10, 20])
        result = payoff_function_dividend(self.S, self.K, 'call')
        npt.assert_array_equal(result, expected)
    
    def test_put_payoff(self):
        """Test put option payoff"""
        expected = np.array([20, 10, 0, 0, 0])
        result = payoff_function_dividend(self.S, self.K, 'put')
        npt.assert_array_equal(result, expected)
    
    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError"""
        with self.assertRaises(ValueError):
            payoff_function_dividend(self.S, self.K, 'invalid')


class TestBoundaryConditionsDividend(unittest.TestCase):
    """Test the boundary_conditions_dividend function"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = DividendBlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
            dividend_yield=0.02
        )
        self.S = np.linspace(0, self.config.S_max, 100)
    
    def test_call_boundary_conditions(self):
        """Test boundary conditions for call option with dividend yield"""
        self.config.option_type = 'call'
        
        # At t=0 (start)
        lower_bc, upper_bc = boundary_conditions_dividend(self.S, 0.0, self.config)
        self.assertEqual(lower_bc, 0.0)
        self.assertGreaterEqual(upper_bc, 0.0)  # Should be non-negative
        
        # At t=T (expiry)
        lower_bc, upper_bc = boundary_conditions_dividend(self.S, self.config.T, self.config)
        self.assertEqual(lower_bc, 0.0)
        expected_upper = self.config.S_max - self.config.K
        self.assertAlmostEqual(upper_bc, expected_upper, places=6)
    
    def test_put_boundary_conditions(self):
        """Test boundary conditions for put option with dividend yield"""
        self.config.option_type = 'put'
        
        # At t=0 (start)
        lower_bc, upper_bc = boundary_conditions_dividend(self.S, 0.0, self.config)
        expected_lower = self.config.K * np.exp(-self.config.r * self.config.T)
        self.assertAlmostEqual(lower_bc, expected_lower, places=6)
        self.assertEqual(upper_bc, 0.0)
    
    def test_discrete_dividend_boundary_adjustment(self):
        """Test boundary conditions with discrete dividends"""
        dividends = [DividendEvent(time=0.5, amount=2.0)]
        config = DividendBlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
            option_type='call', discrete_dividends=dividends
        )
        
        # At t=0.3 (before dividend)
        lower_bc, upper_bc = boundary_conditions_dividend(self.S, 0.3, config)
        self.assertEqual(lower_bc, 0.0)
        self.assertGreater(upper_bc, 0.0)
        
        # At t=0.7 (after dividend)
        lower_bc_after, upper_bc_after = boundary_conditions_dividend(self.S, 0.7, config)
        self.assertEqual(lower_bc_after, 0.0)
        self.assertGreater(upper_bc_after, 0.0)


class TestStockPriceAdjustment(unittest.TestCase):
    """Test dividend-related stock price adjustments"""
    
    def test_adjust_stock_price_for_dividends(self):
        """Test stock price adjustment for dividend payment"""
        S_grid = np.array([95, 100, 105, 110])
        dividend_amount = 2.0
        
        result = adjust_stock_price_for_dividends(S_grid, dividend_amount)
        expected = np.array([93, 98, 103, 108])
        
        npt.assert_array_equal(result, expected)
    
    def test_adjust_stock_price_prevents_negative(self):
        """Test that stock price adjustment prevents negative prices"""
        S_grid = np.array([0.5, 1.0, 2.0, 3.0])
        dividend_amount = 2.5
        
        result = adjust_stock_price_for_dividends(S_grid, dividend_amount)
        expected = np.array([0, 0, 0, 0.5])
        
        npt.assert_array_equal(result, expected)
    
    def test_interpolate_option_values(self):
        """Test option value interpolation"""
        S_old = np.array([90, 100, 110, 120])
        V_old = np.array([10, 20, 30, 40])
        S_new = np.array([95, 105, 115])
        
        result = interpolate_option_values(S_old, V_old, S_new)
        expected = np.array([15, 25, 35])  # Linear interpolation
        
        npt.assert_array_almost_equal(result, expected)
    
    def test_interpolate_option_values_extrapolation(self):
        """Test option value interpolation with extrapolation"""
        S_old = np.array([100, 110, 120])
        V_old = np.array([20, 30, 40])
        S_new = np.array([90, 125])  # Outside the range
        
        result = interpolate_option_values(S_old, V_old, S_new)
        
        # Should use left and right values for extrapolation
        self.assertEqual(result[0], 20)  # Left extrapolation
        self.assertEqual(result[1], 40)  # Right extrapolation


class TestAnalyticalDividendBlackScholes(unittest.TestCase):
    """Test the analytical_dividend_black_scholes function"""
    
    def test_continuous_dividend_call(self):
        """Test call option with continuous dividend yield"""
        S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0.03
        
        result = analytical_dividend_black_scholes(
            S, K, T, r, sigma, q, None, 'call'
        )
        
        # Should be less than non-dividend case
        from black_scholes_solver import analytical_black_scholes
        no_div_result = analytical_black_scholes(S, K, T, r, sigma, 'call')
        
        self.assertLess(result, no_div_result)
        self.assertGreater(result, 0)
    
    def test_continuous_dividend_put(self):
        """Test put option with continuous dividend yield"""
        S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0.03
        
        result = analytical_dividend_black_scholes(
            S, K, T, r, sigma, q, None, 'put'
        )
        
        # Should be greater than non-dividend case
        from black_scholes_solver import analytical_black_scholes
        no_div_result = analytical_black_scholes(S, K, T, r, sigma, 'put')
        
        self.assertGreater(result, no_div_result)
        self.assertGreater(result, 0)
    
    def test_discrete_dividend_adjustment(self):
        """Test option pricing with discrete dividends"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        dividends = [DividendEvent(time=0.5, amount=2.0)]
        
        result = analytical_dividend_black_scholes(
            S, K, T, r, sigma, 0.0, dividends, 'call'
        )
        
        # Should be less than no-dividend case due to PV of dividends
        from black_scholes_solver import analytical_black_scholes
        no_div_result = analytical_black_scholes(S, K, T, r, sigma, 'call')
        
        self.assertLess(result, no_div_result)
        self.assertGreater(result, 0)
    
    def test_multiple_discrete_dividends(self):
        """Test with multiple discrete dividends"""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        dividends = [
            DividendEvent(time=0.25, amount=1.0),
            DividendEvent(time=0.75, amount=1.0)
        ]
        
        result = analytical_dividend_black_scholes(
            S, K, T, r, sigma, 0.0, dividends, 'call'
        )
        
        # Should be less than single dividend case
        single_div = [DividendEvent(time=0.5, amount=1.0)]
        single_result = analytical_dividend_black_scholes(
            S, K, T, r, sigma, 0.0, single_div, 'call'
        )
        
        self.assertLess(result, single_result)
    
    def test_combined_continuous_and_discrete_dividends(self):
        """Test with both continuous yield and discrete dividends"""
        S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0.02
        dividends = [DividendEvent(time=0.5, amount=1.0)]
        
        result = analytical_dividend_black_scholes(
            S, K, T, r, sigma, q, dividends, 'call'
        )
        
        # Should be less than continuous-only case
        continuous_only = analytical_dividend_black_scholes(
            S, K, T, r, sigma, q, None, 'call'
        )
        
        self.assertLess(result, continuous_only)
        self.assertGreater(result, 0)


class TestDividendCrankNicolsonSolver(unittest.TestCase):
    """Test the dividend_crank_nicolson_solver function"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = DividendBlackScholesConfig(
            S_max=150.0, K=100.0, T=0.5, r=0.05, sigma=0.2,
            option_type='call', dividend_yield=0.02
        )
    
    def test_solver_output_shapes(self):
        """Test that solver returns correct output shapes"""
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid, div_indices = dividend_crank_nicolson_solver(
                self.config, N_S, N_t
            )
        
        self.assertEqual(len(S_grid), N_S)
        self.assertEqual(len(t_grid), N_t)
        self.assertEqual(V_grid.shape, (N_S, N_t))
        self.assertIsInstance(div_indices, list)
    
    def test_initial_condition(self):
        """Test that initial condition (payoff) is correctly set"""
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid, div_indices = dividend_crank_nicolson_solver(
                self.config, N_S, N_t
            )
        
        # Final time values should equal payoff
        expected_payoff = payoff_function_dividend(S_grid, self.config.K, self.config.option_type)
        npt.assert_array_almost_equal(V_grid[:, -1], expected_payoff, decimal=10)
    
    def test_discrete_dividend_processing(self):
        """Test processing of discrete dividends"""
        dividends = [DividendEvent(time=0.25, amount=1.0)]
        config = DividendBlackScholesConfig(
            S_max=150.0, K=100.0, T=0.5, r=0.05, sigma=0.2,
            option_type='call', discrete_dividends=dividends
        )
        
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid, div_indices = dividend_crank_nicolson_solver(
                config, N_S, N_t
            )
        
        # Should have identified dividend indices
        self.assertGreater(len(div_indices), 0)
        
        # Check that dividend index is reasonable
        div_time = dividends[0].time
        expected_idx = np.argmin(np.abs(t_grid - div_time))
        self.assertIn(expected_idx, div_indices)
    
    def test_convergence_to_analytical_continuous_dividend(self):
        """Test convergence to analytical solution with continuous dividends"""
        config = DividendBlackScholesConfig(
            S_max=200.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
            option_type='call', dividend_yield=0.03
        )
        
        N_S, N_t = 80, 200  # Moderate resolution for testing
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            S_grid, t_grid, V_grid = dividend_crank_nicolson_solver(config, N_S, N_t)[:3]
        
        # Test at middle stock price
        S_test = 100.0
        S_idx = np.argmin(np.abs(S_grid - S_test))
        numerical_value = V_grid[S_idx, 0]
        
        analytical_value = analytical_dividend_black_scholes(
            S_test, config.K, config.T, config.r, config.sigma,
            config.dividend_yield, config.discrete_dividends, config.option_type
        )
        
        relative_error = abs(numerical_value - analytical_value) / analytical_value
        self.assertLess(relative_error, 0.15)  # Allow reasonable numerical error
    
    def test_dividend_yield_impact(self):
        """Test that dividend yield has expected impact on option values"""
        # Compare options with and without dividend yield
        config_no_div = DividendBlackScholesConfig(
            S_max=150.0, K=100.0, T=0.5, r=0.05, sigma=0.2,
            option_type='call', dividend_yield=0.0
        )
        
        config_with_div = DividendBlackScholesConfig(
            S_max=150.0, K=100.0, T=0.5, r=0.05, sigma=0.2,
            option_type='call', dividend_yield=0.03
        )
        
        N_S, N_t = 30, 50
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            _, _, V_no_div = dividend_crank_nicolson_solver(config_no_div, N_S, N_t)[:3]
            _, _, V_with_div = dividend_crank_nicolson_solver(config_with_div, N_S, N_t)[:3]
        
        # Call option values should be lower with dividend yield
        S_idx = N_S // 2  # Middle stock price
        self.assertLess(V_with_div[S_idx, 0], V_no_div[S_idx, 0])


class TestDividendPlottingFunctions(unittest.TestCase):
    """Test dividend option plotting functions"""
    
    def setUp(self):
        """Set up test data"""
        self.config = DividendBlackScholesConfig(
            S_max=100.0, K=50.0, T=0.25, r=0.05, sigma=0.2,
            option_type='call', dividend_yield=0.02
        )
        
        # Use small grids for fast testing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = dividend_crank_nicolson_solver(self.config, N_S=15, N_t=20)
            self.S_grid, self.t_grid, self.V_grid, self.div_indices = result
    
    def test_plot_dividend_results_execution(self):
        """Test that plot_dividend_results executes without error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                
                plot_dividend_results(
                    self.S_grid, self.t_grid, self.V_grid,
                    self.config, self.div_indices
                )
                
                self.assertTrue(True)  # If we get here, no exception was raised
                
            except Exception as e:
                self.fail(f"plot_dividend_results raised an exception: {e}")
    
    def test_demonstrate_dividend_solver_execution(self):
        """Test that demonstrate_dividend_solver executes without error"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import matplotlib
                matplotlib.use('Agg')
                
                result = demonstrate_dividend_solver()
                
                # Should return tuple of two results
                self.assertEqual(len(result), 2)
                continuous_result, discrete_result = result
                
                # Each result should be a tuple
                self.assertEqual(len(continuous_result), 4)
                self.assertEqual(len(discrete_result), 4)
                
            except Exception as e:
                self.fail(f"demonstrate_dividend_solver raised an exception: {e}")


class TestDividendEdgeCases(unittest.TestCase):
    """Test edge cases for dividend options"""
    
    def test_zero_dividend_yield(self):
        """Test with zero dividend yield (should match basic Black-Scholes)"""
        config = DividendBlackScholesConfig(
            dividend_yield=0.0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                dividend_crank_nicolson_solver(config, N_S=20, N_t=30)
                self.assertTrue(True)  # Should not raise exception
                
            except Exception as e:
                self.fail(f"Zero dividend yield case failed: {e}")
    
    def test_high_dividend_yield(self):
        """Test with high dividend yield"""
        config = DividendBlackScholesConfig(
            dividend_yield=0.15  # 15% dividend yield
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                S_grid, t_grid, V_grid = dividend_crank_nicolson_solver(config, N_S=20, N_t=30)[:3]
                
                # Values should still be reasonable (non-negative for calls, within tolerance)
                if config.option_type == 'call':
                    # Allow small negative values due to numerical precision
                    min_value = np.min(V_grid)
                    self.assertGreater(min_value, -0.1)  # Allow reasonable numerical error
                
            except Exception as e:
                self.fail(f"High dividend yield case failed: {e}")
    
    def test_many_small_dividends(self):
        """Test with many small discrete dividends"""
        dividends = [DividendEvent(time=i*0.1, amount=0.1) for i in range(1, 10)]
        config = DividendBlackScholesConfig(
            T=1.0,
            discrete_dividends=dividends
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                result = dividend_crank_nicolson_solver(config, N_S=25, N_t=50)
                S_grid, t_grid, V_grid, div_indices = result
                
                # Should have processed multiple dividend events
                self.assertGreater(len(div_indices), 0)
                
            except Exception as e:
                self.fail(f"Many dividends case failed: {e}")
    
    def test_large_discrete_dividend(self):
        """Test with large discrete dividend relative to stock price"""
        dividends = [DividendEvent(time=0.5, amount=20.0)]  # Large dividend
        config = DividendBlackScholesConfig(
            S_max=100.0,
            K=50.0,
            discrete_dividends=dividends
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                result = dividend_crank_nicolson_solver(config, N_S=20, N_t=30)
                self.assertTrue(True)  # Should handle large dividends gracefully
                
            except Exception as e:
                self.fail(f"Large dividend case failed: {e}")


if __name__ == '__main__':
    # Set matplotlib backend for testing
    import matplotlib
    matplotlib.use('Agg')
    
    # Run all tests
    unittest.main(verbosity=2)
