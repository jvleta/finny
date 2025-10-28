#!/usr/bin/env python3
"""
Market data driven Black-Scholes examples for AAPL options.

This script mirrors the style of `examples.py`, but it sources option
parameters from live (or most recently available) market data via
`yfinance`. It demonstrates how to download an option chain for AAPL,
convert the records into `BlackScholesConfig` instances, and compare the
numerical PDE solver output against observed market prices.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from black_scholes_solver import BlackScholesConfig, crank_nicolson_solver

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "yfinance is required for market-driven examples. "
        "Install it with `pip install yfinance`."
    ) from exc


MIN_TIME_TO_EXPIRY = 1.0 / 365.0  # Avoid zero-maturity degeneracy


@dataclass
class OptionChain:
    """Container for a single expiration's call and put DataFrames."""

    calls: pd.DataFrame
    puts: pd.DataFrame


@dataclass
class MarketSnapshot:
    """Represents the state of the market data used for the examples."""

    ticker: str
    as_of: datetime
    spot_price: float
    risk_free_rate: float
    expirations: Tuple[str, ...]
    chains: Dict[str, OptionChain]


def fetch_risk_free_rate() -> float:
    """
    Approximate the risk-free rate using the 13-week Treasury bill (^IRX).

    Falls back to 4% if the download fails or yields insufficient data.
    """
    try:
        irx = yf.Ticker("^IRX")
        history = irx.history(period="5d")["Close"].dropna()
        if not history.empty:
            # Convert the quoted percentage yield into a decimal rate
            return float(history.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.04


def fetch_market_snapshot(ticker: str = "AAPL", max_expirations: int = 3) -> MarketSnapshot:
    """
    Download underlying price, risk-free rate, and a slice of the option chain.

    Parameters
    ----------
    ticker:
        Equity ticker to analyse. Defaults to AAPL.
    max_expirations:
        Limit the number of expirations pulled to keep runtime manageable.
    """
    instrument = yf.Ticker(ticker)
    as_of = datetime.now(timezone.utc)

    spot_series = instrument.history(period="5d")["Close"].dropna()
    if spot_series.empty:
        raise RuntimeError(f"Unable to obtain recent prices for {ticker}.")
    spot_price = float(spot_series.iloc[-1])

    expirations = tuple(instrument.options[:max_expirations])
    if not expirations:
        raise RuntimeError(f"No listed options found for {ticker}.")

    chains: Dict[str, OptionChain] = {}
    for expiry in expirations:
        option_chain = instrument.option_chain(expiry)
        chains[expiry] = OptionChain(calls=option_chain.calls, puts=option_chain.puts)

    risk_free_rate = fetch_risk_free_rate()

    return MarketSnapshot(
        ticker=ticker,
        as_of=as_of,
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        expirations=expirations,
        chains=chains,
    )


def time_to_expiration(expiration: str, as_of: datetime) -> float:
    """Convert an expiration string (YYYY-MM-DD) into year fraction."""
    expiry_dt = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    delta = (expiry_dt - as_of).total_seconds()
    if delta <= 0:
        return MIN_TIME_TO_EXPIRY
    return max(delta / (365.0 * 24.0 * 3600.0), MIN_TIME_TO_EXPIRY)


def mid_market_price(row: pd.Series) -> Optional[float]:
    """
    Estimate the market option price using bid/ask midpoint or last price.

    Returns `None` if no reasonable value is available.
    """
    bid = row.get("bid")
    ask = row.get("ask")
    last_price = row.get("lastPrice")

    bid_float = float(bid) if pd.notna(bid) else np.nan
    ask_float = float(ask) if pd.notna(ask) else np.nan
    last_float = float(last_price) if pd.notna(last_price) else np.nan

    if not np.isnan(bid_float) and not np.isnan(ask_float) and ask_float > 0:
        return (bid_float + ask_float) / 2.0
    if not np.isnan(last_float) and last_float > 0:
        return last_float
    return None


def solver_price(config: BlackScholesConfig, spot_price: float) -> float:
    """
    Run the Crank-Nicolson solver and interpolate the value at the spot.

    Returns the interpolated option price at t = 0.
    """
    s_grid, _, v_grid = crank_nicolson_solver(config, N_S=180, N_t=400)
    return float(np.interp(spot_price, s_grid, v_grid[:, 0]))


def select_option_row(
    data: pd.DataFrame,
    target_strike: float,
    *,
    require_iv: bool = True,
) -> Optional[pd.Series]:
    """Locate the row nearest to `target_strike`, optionally requiring implied vol."""
    candidates = data.copy()
    if require_iv:
        candidates = candidates[candidates["impliedVolatility"].notna()]
    if candidates.empty:
        return None
    candidates["distance"] = (candidates["strike"] - target_strike).abs()
    return candidates.nsmallest(1, "distance").iloc[0]


def print_header(title: str) -> None:
    """Shared formatting helper for the console output."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def example_atm_call(snapshot: MarketSnapshot) -> None:
    """Use the nearest-term ATM call option to create an example problem."""
    expiration = snapshot.expirations[0]
    chain = snapshot.chains[expiration].calls
    option_row = select_option_row(chain, snapshot.spot_price)
    if option_row is None:
        print("No suitable call option data available for the first expiration.")
        return

    T = time_to_expiration(expiration, snapshot.as_of)
    sigma = float(option_row["impliedVolatility"])
    strike = float(option_row["strike"])
    market_price = mid_market_price(option_row)

    config = BlackScholesConfig(
        S_max=max(snapshot.spot_price * 3.0, strike * 2.0),
        K=strike,
        T=T,
        r=snapshot.risk_free_rate,
        sigma=sigma,
        option_type="call",
    )

    model_price = solver_price(config, snapshot.spot_price)
    intrinsic = max(snapshot.spot_price - strike, 0.0)

    print_header("AAPL Example 1: Near-Term ATM Call (Market vs PDE Model)")
    print(f"As of:        {snapshot.as_of:%Y-%m-%d %H:%M %Z}")
    print(f"Expiration:   {expiration}  (T = {T:.3f} years)")
    print(f"Strike:       ${strike:.2f}")
    print(f"Spot price:   ${snapshot.spot_price:.2f}")
    print(f"Implied vol:  {sigma * 100:.2f}%")
    print(f"Risk-free r:  {snapshot.risk_free_rate * 100:.2f}%")
    if market_price is not None:
        print(f"Market price: ${market_price:.4f}")
        print(f"Model price:  ${model_price:.4f}")
        print(f"Difference:   ${model_price - market_price:.4f}")
    else:
        print("Market price: unavailable (no bid/ask or last trade)")
        print(f"Model price:  ${model_price:.4f}")
    print(f"Intrinsic:    ${intrinsic:.4f}")


def example_strike_sweep(snapshot: MarketSnapshot) -> None:
    """Compare solver prices across several strikes for the first expiration."""
    expiration = snapshot.expirations[0]
    chain = snapshot.chains[expiration].calls

    calls = chain[chain["impliedVolatility"].notna()].copy()
    if calls.empty:
        print("No call options with implied volatility for strike sweep.")
        return

    calls["distance"] = (calls["strike"] - snapshot.spot_price).abs()
    nearest = calls.nsmallest(5, "distance").sort_values("strike")

    print_header("AAPL Example 2: Strike Sweep for First Expiration (Calls)")
    print(f"Expiration: {expiration}  |  Spot: ${snapshot.spot_price:.2f}")
    print("Strike   IV(%)    Market($)   Model($)    Model-Market($)")
    print("-" * 68)

    T = time_to_expiration(expiration, snapshot.as_of)
    for _, row in nearest.iterrows():
        strike = float(row["strike"])
        sigma = float(row["impliedVolatility"])
        market_price = mid_market_price(row)

        config = BlackScholesConfig(
            S_max=max(snapshot.spot_price * 3.0, strike * 2.0),
            K=strike,
            T=T,
            r=snapshot.risk_free_rate,
            sigma=sigma,
            option_type="call",
        )

        model_price = solver_price(config, snapshot.spot_price)
        diff = model_price - market_price if market_price is not None else float("nan")

        market_str = f"{market_price:9.4f}" if market_price is not None else "    n/a  "
        print(
            f"{strike:6.2f}  {sigma * 100:6.2f}  {market_str}  {model_price:9.4f}  {diff:15.4f}"
        )


def example_term_structure(snapshot: MarketSnapshot) -> None:
    """Hold strike fixed and observe price differences across expirations."""
    base_expiration = snapshot.expirations[0]
    base_chain = snapshot.chains[base_expiration].calls
    base_option = select_option_row(base_chain, snapshot.spot_price)
    if base_option is None:
        print("Cannot determine a reference strike for the term structure example.")
        return

    strike_target = float(base_option["strike"])

    print_header("AAPL Example 3: Term Structure at Fixed Strike (Calls)")
    print(f"Reference strike: ${strike_target:.2f}  |  Spot: ${snapshot.spot_price:.2f}")
    print("Expiration      T(years)   IV(%)    Market($)   Model($)    Model-Market($)")
    print("-" * 86)

    for expiration in snapshot.expirations:
        chain = snapshot.chains[expiration].calls
        option_row = select_option_row(chain, strike_target)
        if option_row is None:
            print(f"{expiration:>12}   --       --        --         --          --")
            continue

        T = time_to_expiration(expiration, snapshot.as_of)
        sigma = float(option_row["impliedVolatility"])
        market_price = mid_market_price(option_row)

        config = BlackScholesConfig(
            S_max=max(snapshot.spot_price * 3.0, strike_target * 2.0),
            K=strike_target,
            T=T,
            r=snapshot.risk_free_rate,
            sigma=sigma,
            option_type="call",
        )

        model_price = solver_price(config, snapshot.spot_price)
        diff = model_price - market_price if market_price is not None else float("nan")
        market_str = f"{market_price:9.4f}" if market_price is not None else "    n/a  "

        print(
            f"{expiration:>12}  {T:8.3f}  {sigma * 100:6.2f}  "
            f"{market_str}  {model_price:9.4f}  {diff:13.4f}"
        )


def main() -> None:
    """Entry point for running all examples."""
    snapshot = fetch_market_snapshot()

    print("=" * 80)
    print("AAPL Market-Driven Black-Scholes PDE Examples")
    print("=" * 80)
    print(f"Data pulled for {snapshot.ticker} at {snapshot.as_of:%Y-%m-%d %H:%M %Z}")
    print(f"Spot price:   ${snapshot.spot_price:.2f}")
    print(f"Risk-free r:  {snapshot.risk_free_rate * 100:.2f}% (approx. from ^IRX)")
    print(f"Expirations:  {', '.join(snapshot.expirations)}")

    example_atm_call(snapshot)
    example_strike_sweep(snapshot)
    example_term_structure(snapshot)

    print("\n" + "=" * 80)
    print("Market-driven examples completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
