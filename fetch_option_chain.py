# fetch_option_chain.py

import pandas as pd
import yfinance as yf
from datetime import datetime


def fetch_option_chain(symbol: str, filter_expiries: list = []) -> pd.DataFrame:
    """
    Fetch the option chain for the given underlying symbol using yfinance.
    Returns a DataFrame with strike, expiration, option_type, last_price, bid, ask,
    implied_volatility, open_interest, volume, T_years (time to expiry).
    If filter_expiries is provided, only those expiry dates will be used.
    """
    tk = yf.Ticker(symbol)
    expiries = list(tk.options)
    if filter_expiries is not []:
        expiries = [e for e in expiries if e in filter_expiries]

    rows = []
    today = datetime.now().date()

    for exp in expiries:
        try:
            opt = tk.option_chain(exp)
        except Exception as e:
            # log/fallback if expiry chain fails
            print(
                f"Warning: could not fetch option chain for {symbol} expiry {exp}: {e}"
            )
            continue

        for df, option_type in [(opt.calls, "call"), (opt.puts, "put")]:
            # add expiration, symbol, type
            df2 = df.copy()
            df2["expiration"] = pd.to_datetime(exp).date()
            df2["option_type"] = option_type
            df2["symbol"] = symbol
            df2["T_years"] = df2["expiration"].apply(lambda d: (d - today).days / 365.0)
            rows.append(df2)

    if not rows:
        return pd.DataFrame()  # or raise

    df_all = pd.concat(rows, ignore_index=True)
    # Keep only needed columns, rename as needed
    keep_cols = [
        "symbol",
        "option_type",
        "expiration",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "T_years",
    ]
    existing = [c for c in keep_cols if c in df_all.columns]
    df_result = df_all[existing].copy()
    # Rename columns for your pipeline standard
    df_result.rename(
        columns={
            "lastPrice": "market_price",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_vol",
        },
        inplace=True,
    )

    return df_result


if __name__ == "__main__":
    symbol = "AAPL"
    df_chain = fetch_option_chain(symbol)
    print(df_chain.head())
