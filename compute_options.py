import os
import glob
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import norm

OUTPUT_DIR = "option_data"
RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0

def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str):
    """Returns (delta, gamma, vega, theta) using Black-Scholes."""
    if not np.isfinite(S) or not np.isfinite(K) or not np.isfinite(T) or not np.isfinite(sigma):
        return np.nan, np.nan, np.nan, np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan, np.nan, np.nan, np.nan

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)

    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)

    # Gamma and Vega are the same for Calls and Puts
    gamma = (exp_qT * n_d1) / (S * sigma * sqrtT)
    vega = (S * exp_qT * n_d1 * sqrtT) / 100.0  # Common to report per 1% vol change

    if option_type.lower() == "call":
        delta = exp_qT * N_d1
        theta = (-(S * sigma * exp_qT * n_d1) / (2 * sqrtT) 
                 - r * K * exp_rT * N_d2 
                 + q * S * exp_qT * N_d1)
    else:
        delta = -exp_qT * N_minus_d1
        theta = (-(S * sigma * exp_qT * n_d1) / (2 * sqrtT) 
                 + r * K * exp_rT * N_minus_d2 
                 - q * S * exp_qT * N_minus_d1)
    
    theta = theta / 365.0  # Report per day

    return float(delta), float(gamma), float(vega), float(theta)

def moneyness_bucket_row(option_type, m):
    if pd.isna(m) or option_type not in ("call", "put"):
        return np.nan

    if option_type == "call":
        if m < 0.85: return "Deep ITM"
        elif m < 0.95: return "ITM"
        elif m <= 1.05: return "ATM"
        elif m <= 1.15: return "OTM"
        else: return "Deep OTM"

    if m > 1.15: return "Deep ITM"
    elif m > 1.05: return "ITM"
    elif m >= 0.95: return "ATM"
    elif m >= 0.85: return "OTM"
    else: return "Deep OTM"

def compute_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # Constants
    out["r"] = RISK_FREE_RATE
    out["q"] = DIVIDEND_YIELD
    
    # ROE and Return
    out["roe"] = out["lastPrice"] / out["underlying_price"] / out["T_years"]
    
    mask_call = out["option_type"] == "call"
    out.loc[mask_call, "tot_return"] = (out["lastPrice"] / out["underlying_price"] / out["T_years"] + 
                                       (out["strike"] / out["underlying_price"]) - 1)
    out.loc[~mask_call, "tot_return"] = out["lastPrice"] / out["underlying_price"] / out["T_years"]
    
    # Moneyness
    out["moneyness"] = out["strike"] / out["underlying_price"]
    out["moneyness_bucket"] = out.apply(
        lambda r: moneyness_bucket_row(r["option_type"], r["moneyness"]), axis=1
    )
    
    # Flags
    out["low_premium_flag"] = out["lastPrice"].abs() < 1.0
    out["far_tenor_flag"] = out["T_years"] > 0.5
    
    # Greeks
    print("Computing Greeks (Delta, Gamma, Vega, Theta)...")
    greeks = out.apply(
        lambda row: bs_greeks(
            S=float(row["underlying_price"]),
            K=float(row["strike"]),
            T=float(row["T_years"]),
            r=float(row["r"]),
            q=float(row["q"]),
            sigma=float(row["impliedVolatility"]),
            option_type=row["option_type"],
        ),
        axis=1
    )
    out[["delta", "gamma", "vega", "theta"]] = pd.DataFrame(greeks.tolist(), index=out.index)
    
    return out

def main():
    # Find most recent raw file
    raw_files = glob.glob(os.path.join(OUTPUT_DIR, "raw_options_*.csv"))
    if not raw_files:
        print("No raw data files found in option_data/")
        return
    
    latest_raw = max(raw_files, key=os.path.getctime)
    print(f"Loading latest raw data: {latest_raw}")
    df = pd.read_csv(latest_raw)
    
    processed_df = compute_calculations(df)
    
    # Reorder columns for convenience
    front = [
        "snapshot_ts_utc", "symbol", "underlying_price",
        "option_type", "expiration", "T_years",
        "strike", "delta", "gamma", "vega", "theta", "roe", "tot_return", "moneyness", "moneyness_bucket",
        "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"
    ]
    cols = [c for c in front if c in processed_df.columns] + [c for c in processed_df.columns if c not in front]
    processed_df = processed_df[cols]
    
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = os.path.join(OUTPUT_DIR, f"computed_options_{date_tag}.csv")
    processed_df.to_csv(out_path, index=False)
    print(f"Saved computed data: {out_path} ({len(processed_df):,} rows)")

if __name__ == "__main__":
    main()
