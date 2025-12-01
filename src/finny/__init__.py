"""Finny - numerical Black-Scholes solvers and utilities."""

from .black_scholes_solver import (
    BlackScholesConfig,
    analytical_black_scholes,
    crank_nicolson_solver,
)
from .american_black_scholes_solver import (
    AmericanBlackScholesConfig,
    american_crank_nicolson_solver,
)
from .dividend_black_scholes_solver import (
    DividendBlackScholesConfig,
    DividendEvent,
    dividend_crank_nicolson_solver,
)

from . import payoffs

__all__ = [
    "BlackScholesConfig",
    "analytical_black_scholes",
    "crank_nicolson_solver",
    "AmericanBlackScholesConfig",
    "american_crank_nicolson_solver",
    "DividendBlackScholesConfig",
    "DividendEvent",
    "dividend_crank_nicolson_solver",
    "payoffs",
]

__version__ = "0.1.0"
