import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union, cast

FloatArray = npt.NDArray[np.float64]
PayoffCallable = Callable[[npt.ArrayLike], FloatArray]
PayoffSeries = Union[
    Mapping[str, PayoffCallable], Sequence[Tuple[str, PayoffCallable]]
]


def _as_float_array(asset_price: npt.ArrayLike) -> FloatArray:
    return np.asarray(asset_price, dtype=float)


def long_call_payoff(asset_price: npt.ArrayLike, strike: float) -> FloatArray:
    """Long call payoff: max(S - E, 0)."""
    asset_array = _as_float_array(asset_price)
    return np.maximum(asset_array - strike, 0.0)


def long_put_payoff(asset_price: npt.ArrayLike, strike: float) -> FloatArray:
    """Long put payoff: max(E - S, 0)."""
    asset_array = _as_float_array(asset_price)
    return np.maximum(strike - asset_array, 0.0)


def short_call_payoff(asset_price: npt.ArrayLike, strike: float) -> FloatArray:
    """Short call payoff: -max(S - E, 0)."""
    return -long_call_payoff(asset_price, strike)


def short_put_payoff(asset_price: npt.ArrayLike, strike: float) -> FloatArray:
    """Short put payoff: -max(E - S, 0)."""
    return -long_put_payoff(asset_price, strike)


def bull_spread_payoff(
    asset_price: npt.ArrayLike, lower_strike: float, upper_strike: float
) -> FloatArray:
    """Bull spread from long lower-strike call and short higher-strike call."""
    return long_call_payoff(asset_price, lower_strike) - long_call_payoff(
        asset_price, upper_strike
    )


def butterfly_spread_payoff(
    asset_price: npt.ArrayLike, low_strike: float, mid_strike: float, high_strike: float
) -> FloatArray:
    """Butterfly spread from long low/high calls and short twice mid call."""
    return (
        long_call_payoff(asset_price, low_strike)
        - 2 * long_call_payoff(asset_price, mid_strike)
        + long_call_payoff(asset_price, high_strike)
    )

# TODO: Implement straddle payoff diagram and add unit test.
# TODO: Implement strangle payoff diagram and add unit test.
# TODO: Implement covered call payoff diagram and add unit test.
# TODO: Implement protective put payoff diagram and add unit test.
# TODO: Implement collar payoff diagram and add unit test.
# TODO: Implement iron condor payoff diagram and add unit test.
# TODO: Implement iron butterfly payoff diagram and add unit test.
# TODO: Implement calendar spread payoff diagram and add unit test.
# TODO: Implement ratio spread payoff diagram and add unit test.
# TODO: Implement condor payoff diagram and add unit test.

def make_payoff(
    payoff_fn: Callable[..., FloatArray], *args: float, **kwargs: float
) -> PayoffCallable:
    """
    Generic factory: returns a callable that binds payoff params except asset_price.
    Assumes payoff_fn takes asset_price as its first parameter.
    """
    return lambda asset_price: payoff_fn(asset_price, *args, **kwargs)


def plot_payoff(
    payoff: PayoffCallable,
    min_asset_price: float,
    max_asset_price: float,
    xlabel: str = r"$S(T)$",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plot a payoff callable over a range of asset prices."""
    asset_prices = np.linspace(min_asset_price, max_asset_price, 1000)
    payout_values = payoff(asset_prices)

    payoff_name = getattr(payoff, "__name__", payoff.__class__.__name__)
    inferred_title = title or f"{payoff_name} Payoff"

    plt.figure()
    plt.plot(asset_prices, payout_values)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(inferred_title)
    plt.grid()
    plt.show()


def plot_payoffs(
    payoffs: PayoffSeries,
    min_asset_price: float,
    max_asset_price: float,
    xlabel: str = r"$S(T)$",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot multiple payoff callables over a shared asset price range.

    Example:
        payoffs = {
            "call": make_payoff(long_call_payoff, strike=100),
            "put": make_payoff(long_put_payoff, strike=100),
        }
        plot_payoffs(payoffs, min_asset_price=80, max_asset_price=120)
    """
    asset_prices = np.linspace(min_asset_price, max_asset_price, 1000)
    payoff_items: Sequence[Tuple[str, PayoffCallable]]
    if isinstance(payoffs, Mapping):
        payoff_items = list(payoffs.items())
    else:
        payoff_items = list(cast(Sequence[Tuple[str, PayoffCallable]], payoffs))

    plt.figure()
    for label, payoff_fn in payoff_items:
        plt.plot(asset_prices, payoff_fn(asset_prices), label=label)

    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title or "Payoffs")
    plt.grid()
    plt.legend()
    plt.show()


# TODO: Add docstrings with examples for all functions.

if __name__ == "__main__":
    # Long call payoff example
    call_payoff_example = make_payoff(long_call_payoff, strike=95)
    plot_payoff(
        call_payoff_example,
        min_asset_price=80,
        max_asset_price=120,
        ylabel=r"$C$",
        title="Call Option Payoff",
    )

    # Long put payoff example
    put_payoff_example = make_payoff(long_put_payoff, strike=105)
    plot_payoff(
        put_payoff_example,
        min_asset_price=80,
        max_asset_price=120,
        ylabel=r"$P$",
        title="Put Option Payoff",
    )

    # Bull spread payoff example
    bull_spread = make_payoff(bull_spread_payoff, lower_strike=90, upper_strike=110)
    plot_payoff(
        bull_spread,
        min_asset_price=80,
        max_asset_price=120,
        title="Bull Spread Payoff",
    )

    # Butterfly spread payoff example
    butterfly_spread = make_payoff(
        butterfly_spread_payoff, low_strike=90, mid_strike=100, high_strike=110
    )
    plot_payoff(
        butterfly_spread,
        min_asset_price=80,
        max_asset_price=120,
        title="Butterfly Spread Payoff",
    )

    # Combined payoffs example
    combined_payoffs = {
        "Long Call (K=95)": call_payoff_example,
        "Long Put (K=105)": put_payoff_example,
        "Bull Spread (K1=90, K2=110)": bull_spread,
        "Butterfly Spread (K1=90, K2=100, K3=110)": butterfly_spread,
    }
    plot_payoffs(
        combined_payoffs,
        min_asset_price=80,
        max_asset_price=120,
        title="Combined Payoffs",
    )
