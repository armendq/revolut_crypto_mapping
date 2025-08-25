"""
Analyses runner (Phase-1 relaxed filters + momentum path)

Outputs to:
  public_runs/latest/summary.json
  public_runs/latest/signals.json
  public_runs/latest/run_stats.json
  public_runs/latest/market_snapshot.json
  public_runs/latest/debug_scan.json

Notes
- Prefers data/portfolio.json for equity/cash; falls back to env vars.
- Universe keeps CG-only assets (no order book) and flags them.
- Two entry paths:
  1) Aggressive 1m breakout + micro pullback (relaxed)
  2) Momentum: 5m HH breakout + 15m ROC + RVOL(5m)
- Always writes candidates even if signal type is "C".
Rules respected:
- DOT is staked/untouchable; ETH & DOT are not rotated.
- Floor check: equity must be >= 40_000 (hard stop to open new risk).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import requests
import pandas as pd

# ---------- Config (relaxed) ----------
VOL_USD_MIN = 3_000_000         # was 8M
MAX_SPREAD = 0.01               # 1% (was 0.5%)
FLOOR_EQUITY = 40_000

# Aggressive 1m breakout window (relaxed)
BREAKOUT_PCT_MIN = 0.010        # +1.0%
BREAKOUT_PCT_MAX = 0.060        # +6.0%
RVOL_1M_MIN = 2.5               # was 4.0
MICRO_FADE_MAX = 0.010          # 1.0% (was 0.6%)

# Momentum path
HH_LOOKBACK_5M = 20             # bars
ROC_15M_LOOKBACK = 30           # bars of 30s (we'll calculate differently)
ROC_15M_MIN = 0.04              # +4%
RVOL_5M_MIN = 1.8

# Risk model
RISK_PCT = 0.012
CAP_OK = 0.60
CAP_WEAK = 0.30

DATA_DIR = "public_runs/latest"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Small helpers ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default

# ---------- Basic market fetchers (CoinGecko + Binance lite) ----------
CG_BASE = "https://api.coingecko.com/api/v3"

def cg_simple_price(ids: List[str]) -> Dict[str, Any]:
    # chunks of 100
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i+100]
        params = {
            "ids": ",".join(chunk),
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_last_updated_at": "true",
        }
        r = requests.get(f"{CG_BASE}/simple/price", params=params, timeout=20)
        r.raise_for_status()
        out.update(r.json())
        time.sleep(0.3)
    return out

def cg_market_chart(coin_id: str, days: int = 1) -> pd.DataFrame:
    # Minute-ish resolution for 1 day
    r = requests.get(f"{CG_BASE}/coins/{coin_id}/market_chart",
                     params={"vs_currency": "usd", "days": days}, timeout=30)
    r.raise_for_status()
    js = r.json()
    # prices: [ [ms, price], ... ]
    df = pd.DataFrame(js.get("prices", []), columns=["ts_ms", "price"])
    df["time"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns=["ts_ms"]).sort_index()
    # build OHLC from price series (approx)
    df["open"] = df["price"].rolling(2).apply(lambda x: x[0], raw=True)
    df["high"] = df["price"].rolling(2).max()
    df["low"] = df["price"].rolling(2).min()
    df["close"] = df["price"]
    df["vol"] = 1.0  # CG doesn't provide minute vol here; will proxy with RVOL from price moves
    return df.dropna()

# ---------- Portfolio inputs ----------
def load_portfolio() -> Dict[str, float]:
    # Try data/portfolio.json first
    pf = read_json("data/portfolio.json")
    if pf and "equity" in pf and "cash" in pf:
        return {"equity": float(pf["equity"]), "cash": float(pf["cash"])}
    # fall back to env
    return {
        "equity": env_float("EQUITY", 41000.0),
        "cash": env_float("CASH", 32000.0),
    }

# ---------- Mapping ----------
def load_mapping() -> pd.DataFrame:
    # Expect mapping at data/mapping.csv or json; otherwise fallback to a tiny universe (ETH/DOT)
    if os.path.exists("data/mapping.csv"):
        df = pd.read_csv("data/mapping.csv")
    elif os.path.exists("data/mapping.json"):
        df = pd.read_json("data/mapping.json")
    else:
        df = pd.DataFrame([
            {"symbol": "ETH", "cg_id": "ethereum"},
            {"symbol": "DOT", "cg_id": "polkadot"},
        ])
    # require cg_id
    df = df[df["cg_id"].notna()].copy()
    df["symbol"] = df["symbol"].str.upper()
    return df

# ---------- Indicators ----------
def atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < n + 1:
        return float("nan")
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]

def rvol(series: pd.Series, lookback: int = 15) -> float:
    if len(series) < lookback + 1:
        return float("nan")
    last = series.iloc[-1]
    avg = series.iloc[-(lookback+1):-1].mean()
    return (last / avg) if avg > 0 else float("nan")

def percent_change(a: float, b: float) -> float:
    if b == 0 or not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return (a - b) / b

# ---------- Entry logic ----------
def aggressive_breakout(df_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df_1m) < 20:
        return None
    c0 = df_1m["close"].iloc[-1]
    c1 = df_1m["close"].iloc[-2]
    pct = percent_change(c0, c1)
    if not (BREAKOUT_PCT_MIN <= pct <= BREAKOUT_PCT_MAX):
        return None
    # Micro pullback: last bar low within MICRO_FADE_MAX of close
    low_last = df_1m["low"].iloc[-1]
    if (c0 - low_last) / c0 > MICRO_FADE_MAX:
        return None
    # RVOL on 1m close change magnitude
    change_mag = df_1m["close"].diff().abs().fillna(0.0)
    rv = rvol(change_mag, 15)
    if not (rv >= RVOL_1M_MIN):
        return None
    # Break 15-bar high
    hh = df_1m["high"].rolling(15).max().iloc[-2]
    if c0 <= hh:
        return None
    return {"entry": float(c0), "stop_hint": float(low_last), "atype": "aggr_1m"}

def momentum_breakout(df_5m: pd.DataFrame, df_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df_5m) < HH_LOOKBACK_5M + 1 or len(df_1m) < 30:
        return None
    c0 = df_5m["close"].iloc[-1]
    hh = df_5m["high"].rolling(HH_LOOKBACK_5M).max().iloc[-2]
    if not (c0 > hh):
        return None
    # 15m ROC via 1m series
    c1m = df_1m["close"]
    roc = percent_change(c1m.iloc[-1], c1m.iloc[max(0, len(c1m) - 15):].iloc[0])
    if not (roc >= ROC_15M_MIN):
        return None
    # 5m RVOL using absolute returns
    change_mag = df_5m["close"].pct_change().abs().fillna(0.0)
    rv = rvol(change_mag, 12)  # ~1 hour of 5m bars for mean
    if not (rv >= RVOL_5M_MIN):
        return None
    low_last1m = df_1m["low"].iloc[-1]
    return {"entry": float(c0), "stop_hint": float(low_last1m), "atype": "mom_5m"}

# ---------- Sizing ----------
def position_size_usd(equity: float, cash: float, regime_ok: bool, entry: float, stop: float) -> float:
    cap = CAP_OK if regime_ok else CAP_WEAK
    max_alloc = (equity + cash) * cap
    risk_dollars = equity * RISK_PCT
    per_unit_risk = max(entry - stop, entry * 0.005)
    qty = risk_dollars / per_unit_risk
    usd = min(qty * entry, max_alloc)
    return max(0.0, float(usd))

# ---------- Pipeline ----------
def build_universe(map_df: pd.DataFrame) -> pd.DataFrame:
    ids = map_df["cg_id"].dropna().tolist()
    prices = cg_simple_price(ids) if ids else {}
    rows = []
    for _, row in map_df.iterrows():
        cg_id = row["cg_id"]
        symbol = row["symbol"].upper()
        p = prices.get(cg_id, {})
        price = p.get("usd")
        vol = p.get("usd_24h_vol")
        ok_vol = (vol or 0) >= VOL_USD_MIN
        rows.append({
            "symbol": symbol,
            "cg_id": cg_id,
            "price": price,
            "vol24": vol,
            "spread": None,      # unknown (CG only)
            "src": "coingecko",
            "pass_liq": bool(ok_vol),
            "pass_spread": True, # unknown → allow
        })
    df = pd.DataFrame(rows)
    return df

def fetch_ohlc_cg(cg_id: str) -> Dict[str, pd.DataFrame]:
    # 1-day minute-ish resolution is adequate
    df = cg_market_chart(cg_id, days=1)
    if df.empty:
        return {}
    # Resample to 1m and 5m OHLC (best-effort from price)
    df1 = df[["open", "high", "low", "close"]].copy()
    df1m = df1.resample("1T").last().ffill()
    df5m = df1.resample("5T").last().ffill()
    return {"1m": df1m.tail(200), "5m": df5m.tail(200)}

def compute_regime(btc_rows: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    # BTC regime from 1m series with VWAP proxy (EMA of price as proxy) + EMA9
    out = {"ok": False, "reason": "no-data", "last": None, "vwap": None, "ema9": None}
    if not btc_rows:
        return out
    df1 = btc_rows["1m"]
    if df1 is None or df1.empty:
        return out
    close = df1["close"]
    last = float(close.iloc[-1])
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    vwap_proxy = float(close.ewm(span=30, adjust=False).mean().iloc[-1])
    ok = (last >= vwap_proxy) and (last >= ema9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-or-ema", "last": last, "vwap": vwap_proxy, "ema9": ema9}

def main():
    t0 = time.time()
    portfolio = load_portfolio()
    mapping = load_mapping()

    # Build universe with CG-only (allow if vol >= threshold)
    uni = build_universe(mapping)
    uni_ok = uni[uni["pass_liq"] & uni["pass_spread"]].copy()

    # Always include ETH and DOT even if liq fails (for status/visibility)
    force_keep = {"ETH", "DOT"}
    uni_ok = pd.concat([uni_ok, uni[uni["symbol"].isin(force_keep)]]).drop_duplicates("symbol")

    # Fetch OHLC for BTC + universe
    sym_to_rows: Dict[str, Dict[str, pd.DataFrame]] = {}
    debug_scan = []

    # Include BTC for regime
    btc_rows = fetch_ohlc_cg("bitcoin")
    regime = compute_regime(btc_rows)

    candidates: List[Dict[str, Any]] = []
    signals_out: Dict[str, Any] = {"type": "C", "text": "Hold and wait."}

    for _, r in uni_ok.iterrows():
        sym = r["symbol"]
        if sym == "DOT":
            reason = "dot-staked"
            debug_scan.append({"symbol": sym, "kept": True, "note": reason})
            continue
        rows = fetch_ohlc_cg(r["cg_id"])
        if not rows:
            debug_scan.append({"symbol": sym, "kept": False, "note": "no-ohlc"})
            continue

        df1 = rows["1m"]
        df5 = rows["5m"]
        atr_val = float(atr(df1, 14)) if len(df1) >= 20 else float("nan")
        last = float(df1["close"].iloc[-1])

        # Aggressive breakout
        sig = aggressive_breakout(df1)
        # Momentum fallback
        if sig is None:
            sig = momentum_breakout(df5, df1)

        entry = stop = None
        if sig:
            entry = float(sig["entry"])
            # Use breakout bar low as stop; guard if ATR < noise
            stop_hint = float(sig["stop_hint"])
            min_noise = max(entry * 0.006, atr_val * 0.6 if math.isfinite(atr_val) else entry * 0.006)
            stop = min(stop_hint, entry - min_noise)

        # Candidate row (even if no signal)
        cand = {
            "symbol": sym,
            "src": r["src"],
            "price": last,
            "atr": atr_val if math.isfinite(atr_val) else None,
            "entry": entry,
            "stop": stop,
            "atype": sig["atype"] if sig else None,
            "vol24": r["vol24"],
        }
        candidates.append(cand)
        debug_scan.append({"symbol": sym, "kept": True, "note": "ok" if sig else "no-entry"})

    # Rank candidates (prefer with entry, then RVOL proxy via recent pct change)
    def cand_score(c):
        base = 1.0 if c["entry"] else 0.0
        # favor higher 15m momentum using 1m series if available (approx via (entry/price))
        mom = 0.0
        if c["entry"] and c["price"] and c["price"] > 0:
            mom = min(2.0, (c["entry"] / c["price"]) - 1.0)
        return base * 10 + mom

    candidates_sorted = sorted(candidates, key=cand_score, reverse=True)

    # Select a signal if any candidate has an entry and we pass guards
    chosen = next((c for c in candidates_sorted if c["entry"]), None)

    if chosen:
        # Guard rails: ETH and DOT not rotated (we will not rotate out, but can buy ETH if not rotating? Keep simple: allow ETH buy; do not sell)
        # Floor check for new risk
        if portfolio["equity"] < FLOOR_EQUITY:
            signals_out = {"type": "C", "text": "Equity below floor; hold and wait."}
        else:
            entry = float(chosen["entry"])
            stop = float(chosen["stop"])
            atr_val = float(chosen["atr"]) if chosen["atr"] else entry * 0.01
            t1 = entry + 0.8 * atr_val
            t2 = entry + 1.5 * atr_val
            size_usd = position_size_usd(
                equity=portfolio["equity"],
                cash=portfolio["cash"],
                regime_ok=bool(regime["ok"]),
                entry=entry,
                stop=stop,
            )
            signals_out = {
                "type": "B",
                "text": "Breakout entry.",
                "symbol": chosen["symbol"],
                "entry": round(entry, 6),
                "stop": round(stop, 6),
                "t1": round(t1, 6),
                "t2": round(t2, 6),
                "size_usd": round(size_usd, 2),
                "trail_after_R": 1.0,
                "atype": chosen.get("atype"),
            }

    # Assemble summary
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "regime": regime,
        "equity": float(portfolio["equity"]),
        "cash": float(portfolio["cash"]),
        "candidates": candidates_sorted[:8],  # top few for brevity
        "signals": signals_out,
        "run_stats": {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    # Write outputs atomically
    write_json(os.path.join(DATA_DIR, "summary.json"), summary)
    write_json(os.path.join(DATA_DIR, "signals.json"), {"signals": signals_out})
    write_json(os.path.join(DATA_DIR, "run_stats.json"), summary["run_stats"])
    write_json(os.path.join(DATA_DIR, "market_snapshot.json"), {
        "universe_size": int(len(uni)),
        "universe_kept": int(len(uni_ok)),
    })
    write_json(os.path.join(DATA_DIR, "debug_scan.json"), debug_scan)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Always emit a minimal summary so the workflow doesn't leave stale Pages
        err = {
            "generated_at_utc": utcnow_iso(),
            "regime": {"ok": False, "reason": "exception", "last": None, "vwap": None, "ema9": None},
            "equity": env_float("EQUITY", 41000.0),
            "cash": env_float("CASH", 32000.0),
            "candidates": [],
            "signals": {"type": "C", "text": f"Error: {type(e).__name__}"},
            "run_stats": {"elapsed_ms": 0, "time": utcnow_iso()},
        }
        write_json(os.path.join(DATA_DIR, "summary.json"), err)
        write_json(os.path.join(DATA_DIR, "signals.json"), {"signals": err["signals"]})
        write_json(os.path.join(DATA_DIR, "run_stats.json"), err["run_stats"])
        raise