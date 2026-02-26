import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["NVDA", "TSLA","SPY"]
OUTPUT_DIR = "option_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def yearfrac_to_expiration(expiration_yyyy_mm_dd: str, now_utc: datetime) -> float:
    exp_dt = datetime.strptime(expiration_yyyy_mm_dd, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    seconds = (exp_dt - now_utc).total_seconds()
    return max(seconds, 0.0) / (365.0 * 24 * 3600)

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

def download_full_chain_raw(symbol: str) -> pd.DataFrame | None:
    print(f"\n=== Downloading {symbol} ===")
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

        try:
            chain = t.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()

            for df, opt_type in [(calls, "call"), (puts, "put")]:
                df["symbol"] = symbol
                df["option_type"] = opt_type
                df["expiration"] = exp
                df["snapshot_ts_utc"] = snapshot_ts_utc
                df["underlying_price"] = underlying_price
                df["T_years"] = T
                all_rows.append(df)
        except Exception as e:
            print(f"  Error downloading {exp}: {e}")

    if not all_rows:
        return None
    
    return pd.concat(all_rows, ignore_index=True)

def main():
    all_dfs = []
    for symbol in TICKERS:
        df = download_full_chain_raw(symbol)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No data collected.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = os.path.join(OUTPUT_DIR, f"raw_options_{date_tag}.csv")
    combined_df.to_csv(out_path, index=False)
    print(f"\nSaved raw data: {out_path} ({len(combined_df):,} rows)")

if __name__ == "__main__":
    main()
