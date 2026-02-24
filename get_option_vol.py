import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


TICKERS = ["NVDA", "TSLA"]
TICKERS2 = ["SPY", "TSLA", "NVDA"]

OUTPUT_DIR = "option_data"
RISK_FREE_RATE = 0.045  # set your own; continuous approx
DIVIDEND_YIELD = 0.0    # set if you want (SPY ~ small, but optional)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def yearfrac_to_expiration(expiration_yyyy_mm_dd: str, now_utc: datetime) -> float:
    exp_dt = datetime.strptime(expiration_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    seconds = (exp_dt - now_utc).total_seconds()
    return max(seconds, 0.0) / (365.0 * 24 * 3600)


def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    """Black–Scholes delta with dividend yield q (continuous)."""
    if not np.isfinite(S) or not np.isfinite(K) or not np.isfinite(T) or not np.isfinite(sigma):
        return np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(-np.exp(-q * T) * norm.cdf(-d1))


def get_underlying_price(t: yf.Ticker) -> float:
    # try fast_info first
    try:
        fi = getattr(t, "fast_info", None)
        if fi is not None:
            for key in ["last_price", "lastPrice", "regularMarketPrice"]:
                v = fi.get(key)
                if v is not None and np.isfinite(v):
                    return float(v)
    except Exception:
        pass

    # fallback: 1d history close
    hist = t.history(period="5d", interval="1d", auto_adjust=False)
    if hist is not None and len(hist) > 0:
        return float(hist["Close"].iloc[-1])

    return np.nan

def moneyness_bucket_row(option_type, m):
    if pd.isna(m) or option_type not in ("call", "put"):
        return np.nan

    if option_type == "call":
        if m < 0.80:
            return "Deep ITM"
        elif m < 0.95:
            return "ITM"
        elif m <= 1.05:
            return "ATM"
        elif m <= 1.20:
            return "OTM"
        else:
            return "Deep OTM"

    # put (reversed)
    if m > 1.20:
        return "Deep ITM"
    elif m > 1.05:
        return "ITM"
    elif m >= 0.95:
        return "ATM"
    elif m >= 0.80:
        return "OTM"
    else:
        return "Deep OTM"


def add_common_fields(df: pd.DataFrame, *, symbol: str, option_type: str, expiration: str,
                      snapshot_ts_utc: str, underlying_price: float, T: float) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = symbol
    out["option_type"] = option_type
    out["expiration"] = expiration
    out["snapshot_ts_utc"] = snapshot_ts_utc
    out["underlying_price"] = underlying_price
    out["T_years"] = T
    out["r"] = RISK_FREE_RATE
    out["q"] = DIVIDEND_YIELD
    out["roe"]=out["lastPrice"]/out["underlying_price"]/out["T_years"]
    if option_type == "call":
        out["tot_return"]=out["lastPrice"]/out["underlying_price"]/out["T_years"] + (out["strike"])/out["underlying_price"]-1
    else:
        out["tot_return"]=out["lastPrice"]/out["underlying_price"]/out["T_years"] 
    
    out["moneyness"]=out["strike"]/out["underlying_price"]
    #out["moneyness_bucket"]=out["moneyness"].apply(moneyness_bucket)
    out["moneyness_bucket"] = out.apply(
        lambda r: moneyness_bucket_row(r["option_type"], r["moneyness"]),
        axis=1,
    )
    out["low_premium_flag"]=out["lastPrice"].abs()<1.0
    out["far_tenor_flag"]=out["T_years"]>1.0
    return out


def compute_delta_column(df: pd.DataFrame) -> pd.Series:
    # Yahoo uses impliedVolatility as decimal (e.g. 0.35)
    sigma = pd.to_numeric(df.get("impliedVolatility", np.nan), errors="coerce")
    strike = pd.to_numeric(df.get("strike", np.nan), errors="coerce")

    # vectorized-ish apply (fast enough for chains; can be optimized later)
    return df.apply(
        lambda row: bs_delta(
            S=float(row["underlying_price"]),
            K=float(row["strike"]) if pd.notna(row["strike"]) else np.nan,
            T=float(row["T_years"]),
            r=float(row["r"]),
            q=float(row["q"]),
            sigma=float(row["impliedVolatility"]) if pd.notna(row.get("impliedVolatility")) else np.nan,
            option_type=row["option_type"],
        ),
        axis=1
    )


def download_full_chain_with_delta(symbol: str) -> pd.DataFrame | None:
    print(f"\n=== {symbol} ===")
    t = yf.Ticker(symbol)

    snapshot_ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    underlying_price = get_underlying_price(t)
    print(f"Underlying price snapshot: {underlying_price} @ {snapshot_ts_utc}")

    expirations = t.options
    if not expirations:
        print("No expirations returned.")
        return None

    all_rows = []
    now_utc = datetime.now(timezone.utc)

    for exp in expirations:
        T = yearfrac_to_expiration(exp, now_utc)
        print(f"  Exp {exp} (T={T:.4f}y)")

        chain = t.option_chain(exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls = add_common_fields(
            calls, symbol=symbol, option_type="call", expiration=exp,
            snapshot_ts_utc=snapshot_ts_utc, underlying_price=underlying_price, T=T
        )
        puts = add_common_fields(
            puts, symbol=symbol, option_type="put", expiration=exp,
            snapshot_ts_utc=snapshot_ts_utc, underlying_price=underlying_price, T=T
        )

        # delta
        calls["delta"] = compute_delta_column(calls)
        puts["delta"] = compute_delta_column(puts)

        all_rows.append(calls)
        all_rows.append(puts)

    df = pd.concat(all_rows, ignore_index=True)

    # Helpful ordering (keep any extra Yahoo columns too)
    front = [
        "snapshot_ts_utc", "symbol", "underlying_price",
        "option_type", "expiration", "T_years",
        "strike", "delta",
        "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility",
        "inTheMoney", "contractSymbol", "lastTradeDate",
        "r", "q","roe","moneyness"
    ]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]

    return df


def main():
    all_dfs = []
    for symbol in TICKERS:
        df = download_full_chain_with_delta(symbol)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No data collected.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = os.path.join(OUTPUT_DIR, f"combined_options_with_spot_delta_{date_tag}.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"\nFinal Combined Saved: {out_path} ({len(combined_df):,} rows)")


if __name__ == "__main__":
    main()
