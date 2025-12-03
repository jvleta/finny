import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from finny.payoffs import (
    long_call_payoff,
    long_put_payoff,
    short_call_payoff,
    short_put_payoff,
    bull_spread_payoff,
    butterfly_spread_payoff,
    make_payoff,
    plot_payoff,
    plot_payoffs,
)

matplotlib.use("Agg")


def test_payoff_functions():
    asset_prices = np.array([80, 90, 100, 110, 120], dtype=float)

    call_payoff = make_payoff(long_call_payoff, strike=100)
    put_payoff = make_payoff(long_put_payoff, strike=100)
    short_call = make_payoff(short_call_payoff, strike=100)
    short_put = make_payoff(short_put_payoff, strike=100)
    bull_spread = make_payoff(bull_spread_payoff, lower_strike=90, upper_strike=110)
    butterfly_spread = make_payoff(
        butterfly_spread_payoff, low_strike=90, mid_strike=100, high_strike=110
    )

    np.testing.assert_array_equal(
        call_payoff(asset_prices),
        np.array([0, 0, 0, 10, 20], dtype=float),
    )
    np.testing.assert_array_equal(
        put_payoff(asset_prices),
        np.array([20, 10, 0, 0, 0], dtype=float),
    )
    np.testing.assert_array_equal(
        short_call(asset_prices),
        -call_payoff(asset_prices),
    )
    np.testing.assert_array_equal(
        short_put(asset_prices),
        -put_payoff(asset_prices),
    )

    np.testing.assert_array_equal(
        bull_spread(asset_prices),
        np.array([0, 0, 10, 20, 20], dtype=float),
    )
    np.testing.assert_array_equal(
        butterfly_spread(asset_prices),
        np.array([0, 0, 10, 0, 0], dtype=float),
    )
    np.testing.assert_array_equal(
        butterfly_spread(np.array([105], dtype=float)),
        np.array([5], dtype=float),
    )

    np.testing.assert_array_equal(
        call_payoff(asset_prices),
        np.array([0, 0, 0, 10, 20], dtype=float),
    )


def test_plot_payoff(monkeypatch):
    shown = {"called": False}
    monkeypatch.setattr(plt, "show", lambda: shown.__setitem__("called", True))
    plt.close("all")

    call_payoff = make_payoff(long_call_payoff, strike=100)
    plot_payoff(call_payoff, min_asset_price=80, max_asset_price=120)

    fig = plt.gcf()
    ax = fig.axes[0]
    line = ax.lines[0]

    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)

    assert shown["called"]
    assert len(xdata) == 1000
    np.testing.assert_allclose(ydata, long_call_payoff(xdata, strike=100))


def test_plot_payoffs(monkeypatch):
    shown = {"called": False}
    monkeypatch.setattr(plt, "show", lambda: shown.__setitem__("called", True))
    plt.close("all")

    payoffs = {
        "call": make_payoff(long_call_payoff, strike=100),
        "put": make_payoff(long_put_payoff, strike=100),
    }
    plot_payoffs(payoffs, min_asset_price=80, max_asset_price=120)

    fig = plt.gcf()
    ax = fig.axes[0]
    lines = ax.lines

    assert shown["called"]
    assert len(lines) == 2

    xdata = np.asarray(lines[0].get_xdata(), dtype=float)
    np.testing.assert_allclose(xdata, np.asarray(lines[1].get_xdata(), dtype=float))
    np.testing.assert_allclose(np.asarray(lines[0].get_ydata()), payoffs["call"](xdata))
    np.testing.assert_allclose(np.asarray(lines[1].get_ydata()), payoffs["put"](xdata))
