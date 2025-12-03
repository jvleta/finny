import functools
import numpy as np
import matplotlib.pyplot as plt


def vectorize_payoff(payoff_fn, name=None):
    """
    Vectorize a scalar payoff function using NumPy while preserving metadata.
    """
    if getattr(payoff_fn, "_is_vectorized_payoff", False):
        return payoff_fn

    vectorized = np.vectorize(payoff_fn, otypes=[float])

    def wrapper(*args, **kwargs):
        return vectorized(*args, **kwargs)

    wrapper._is_vectorized_payoff = True  # type: ignore[attr-defined]
    wrapper.__name__ = name or getattr(
        payoff_fn, "__name__", payoff_fn.__class__.__name__
    )
    wrapper.__doc__ = getattr(payoff_fn, "__doc__", None)
    return wrapper


def long_call_payoff(asset_price, strike):
    """Long call payoff: max(S - E, 0)."""
    diff = asset_price - strike
    return diff if diff > 0 else 0.0


def long_put_payoff(asset_price, strike):
    """Long put payoff: max(E - S, 0)."""
    diff = strike - asset_price
    return diff if diff > 0 else 0.0


def short_call_payoff(asset_price, strike):
    """Short call payoff: -max(S - E, 0)."""
    return -long_call_payoff(asset_price, strike)


def short_put_payoff(asset_price, strike):
    """Short put payoff: -max(E - S, 0)."""
    return -long_put_payoff(asset_price, strike)


def bull_spread_payoff(asset_price, lower_strike, upper_strike):
    """Bull spread from long lower-strike call and short higher-strike call."""
    return long_call_payoff(asset_price, lower_strike) - long_call_payoff(
        asset_price, upper_strike
    )


def butterfly_spread_payoff(asset_price, low_strike, mid_strike, high_strike):
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


def make_payoff(payoff_fn, *args, **kwargs):
    """
    Generic factory: returns a callable that binds payoff params except asset_price.
    Assumes payoff_fn takes asset_price as its first parameter.
    """
    payoff_name = getattr(payoff_fn, "__name__", payoff_fn.__class__.__name__)

    if getattr(payoff_fn, "_is_vectorized_payoff", False):

        def payoff(asset_price):
            return payoff_fn(asset_price, *args, **kwargs)

        payoff.__name__ = payoff_name
        payoff.__doc__ = getattr(payoff_fn, "__doc__", None)
        return payoff

    bound_payoff = functools.partial(payoff_fn, *args, **kwargs)
    return vectorize_payoff(bound_payoff, name=payoff_name)


def plot_payoff(
    payoff,
    min_asset_price,
    max_asset_price,
    xlabel=r"$S(T)$",
    ylabel=None,
    title=None,
):
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


def plot_payoffs(payoffs, min_asset_price, max_asset_price):
    """Plot multiple payoff callables over a range of asset prices."""
    asset_prices = np.linspace(min_asset_price, max_asset_price, 1000)

    plt.figure()
    for name, payoff in payoffs.items():
        plt.plot(asset_prices, payoff(asset_prices), label=name)

    plt.xlabel(r"$S(T)$")
    plt.title("Payoffs")
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
    )
