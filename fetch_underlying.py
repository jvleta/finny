import pandas as pd
import yfinance as yf


def get_spot_price(symbol: str) -> float:
    data = yf.Ticker(symbol).history(period="1d")
    if data.empty:
        return 0.0
    return float(data["Close"].iloc[-1])


def get_dividend_yield(symbol: str) -> float:
    dividends = yf.Ticker(symbol).dividends
    if dividends.empty:
        return 0.0

    cutoff = dividends.index.max() - pd.Timedelta(days=365)
    trailing_dividends = dividends[dividends.index >= cutoff].sum()
    price = get_spot_price(symbol)
    return float(trailing_dividends) / price if price else 0.0


def get_risk_free_rate() -> float:
    # 10Y Treasury yield via yfinance ticker ^TNX
    hist = yf.Ticker("^TNX").history(period="5d")
    if hist.empty:
        return 0.0
    rate_pct = hist["Close"].dropna().iloc[-1]
    return float(rate_pct) / 100.0


def fetch_underlying_data(symbol: str) -> dict:
    S = get_spot_price(symbol)
    q = get_dividend_yield(symbol)
    r = get_risk_free_rate()
    return {"symbol": symbol, "spot_price": S, "dividend_yield": q, "risk_free_rate": r}


if __name__ == "__main__":
    symbol = "AAPL"
    data = fetch_underlying_data(symbol)
    print(f"The current spot price of {symbol} is: ${data['spot_price']:.2f}")
    print(f"The trailing dividend yield of {symbol} is: {data['dividend_yield']*100:.2f}%")
    print(f"The current risk-free rate is: {data['risk_free_rate']*100:.2f}%")
