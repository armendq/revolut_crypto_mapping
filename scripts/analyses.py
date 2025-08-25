#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heavy market scanner for Revolut universe with multi-timeframe confirmation.

Outputs (always written):
  public_runs/latest/summary.json
  public_runs/latest/market_snapshot.json
  public_runs/latest/debug_scan.json
  public_runs/latest/run_stats.json
  public_runs/latest/signals.json

Key rules:
- DOT is staked/untouchable; ETH and DOT are not rotated.
- Stop = breakout bar low. Targets: T1=entry+0.8*ATR(5m), T2=entry+1.5*ATR(5m).
- Position sizing from env EQUITY/CASH, RISK_PCT=1.2%, cap 60% if regime.ok else 30%.
- Equity floor = 40000 with buffer guard.

Requires: pandas, requests (install in workflow).
"""

import os
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

# ---------------- Paths
OUT_DIR = Path("public_runs/latest")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True)
DATA = Path("data"); DATA.mkdir(exist_ok=True)

SUMMARY = OUT_DIR / "summary.json"
SNAPSHOT = OUT_DIR / "market_snapshot.json"
DEBUG = OUT_DIR / "debug_scan.json"
RUNSTATS = OUT_DIR / "run_stats.json"
SIGNALS = OUT_DIR / "signals.json"

# ---------------- Config
EQUITY = float(os.getenv("EQUITY", "41000") or "41000")
CASH   = float(os.getenv("CASH", "32000") or "32000")
EQUITY_FLOOR = 40000.0
RISK_PCT = 0.012
CAP_STRONG = 0.60
CAP_WEAK = 0.30

# Binance
BINANCE = "https://data-api.binance.vision"
KLINES = f"{BINANCE}/api/v3/klines"             # ?symbol=BTCUSDT&interval=5m&limit=500
TICKER24 = f"{BINANCE}/api/v3/ticker/24hr"      # ?symbol=BTCUSDT
DEPTH = f"{BINANCE}/api/v3/depth"               # ?symbol=BTCUSDT&limit=5

# Fetch
RETRIES = 3
TIMEOUT = 20
SLEEP_BASE = 0.8

# Filters
MIN_QUOTE_VOL_24H = 3_000_000   # USD 24h quote volume
MAX_SPREAD = 0.006              # <= 0.6%
UNROTATE = {"ETH", "DOT"}
UNTOUCHABLE = {"DOT"}

# Signals / logic
ATR_LEN_5M = 14
EMA_LEN = 20
BREAKOUT_LOOKBACK_5M = 20       # prior 20 highs
RVOL_WINDOW_5M = 20
RVOL_MIN = 3.0
BAR_RET_MIN = 0.012             # +1.2% on breakout bar
BAR_RET_MAX = 0.06              # cap to avoid blow-offs
HTF_CONFIRM_EMA = 20            # 1h EMA
HTF_CONFIRM = True              # require 1h close above EMA & VWAP

VERBOSE = True
def log(*a: Any) -> None:
    if VERBOSE:
        print("[analyses]", *a, flush=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------- HTTP helpers
def http_get_json(url: str, params: Dict[str, Any] = None) -> Optional[Any]:
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent":"rev-scan/1.1"})
            if r.status_code == 200:
                return r.json()
            log(f"GET {url} status {r.status_code} (try {attempt})")
        except Exception as e:
            log(f"GET {url} error {e} (try {attempt})")
        time.sleep(SLEEP_BASE * attempt)
    return None

# ---------------- Market helpers
def get_mapping_universe() -> List[str]:
    # Expect data/revolut_mapping.json listing tickers; robust parse
    path = DATA / "revolut_mapping.json"
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            out: List[str] = []
            if isinstance(obj, list):
                for m in obj:
                    if isinstance(m, dict):
                        t = (m.get("ticker") or m.get("symbol") or "").upper()
                        if t: out.append(t)
                    elif isinstance(m, str):
                        out.append(m.upper())
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, dict):
                        t = (v.get("ticker") or v.get("symbol") or k).upper()
                        if t: out.append(t)
                    else:
                        out.append(str(k).upper())
            return sorted(list(set(out)))
        except Exception as e:
            log("mapping parse error:", e)
    # fallback: top USDT by quote volume
    data = http_get_json(TICKER24)
    syms: List[str] = []
    if isinstance(data, list):
        rows = []
        for d in data:
            s = d.get("symbol", "")
            if s.endswith("USDT") and all(x not in s for x in ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT")):
                qv = float(d.get("quoteVolume", "0") or 0)
                rows.append((s.replace("USDT",""), qv))
        rows.sort(key=lambda x: x[1], reverse=True)
        syms = [t for t,_ in rows[:200]]
    return syms

def fetch_klines(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    q = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
    arr = http_get_json(KLINES, q)
    if not isinstance(arr, list) or not arr:
        return None
    try:
        df = pd.DataFrame(arr, columns=["t","o","h","l","c","v","ct","qv","trades","tbb","tbq","ig"])
        df = df[["t","o","h","l","c","v","qv"]].copy()
        df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        for col in ("o","h","l","c","v","qv"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def fetch_orderbook(symbol: str) -> Optional[Tuple[float,float]]:
    q = {"symbol": f"{symbol}USDT", "limit": 5}
    d = http_get_json(DEPTH, q)
    if not d or "bids" not in d or "asks" not in d:
        return None
    try:
        bid = float(d["bids"][0][0])
        ask = float(d["asks"][0][0])
        return bid, ask
    except Exception:
        return None

def fetch_24h(symbol: str) -> Optional[Dict[str,float]]:
    q = {"symbol": f"{symbol}USDT"}
    d = http_get_json(TICKER24, q)
    if not d:
        return None
    try:
        last = float(d["lastPrice"])
        quote_vol = float(d["quoteVolume"])
        return {"last": last, "quote_vol": quote_vol}
    except Exception:
        return None

# ---------------- TA helpers
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    pv = (tp * df["v"]).cumsum()
    vv = df["v"].cumsum()
    return pv / vv.replace(0, float("nan"))

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["c"].shift(1)
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev_close).abs(),
        (df["l"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rel_volume(df: pd.DataFrame, window: int) -> float:
    if len(df) < window + 1:
        return 0.0
    vol_med = df["v"].iloc[-window-1:-1].median()
    return 0.0 if vol_med <= 0 else float(df["v"].iloc[-1] / vol_med)

# ---------------- Scoring & sizing
def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid

def position_usd(entry: float, stop: float, equity: float, cash: float, strong: bool) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-8)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = CAP_STRONG if strong else CAP_WEAK
    usd_cap = equity * cap
    return max(0.0, min(usd, usd_cap, cash))

# ---------------- Core detection
def candidate_for_symbol(sym: str, debug: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 24h liquidity and spread gate
    stats24 = fetch_24h(sym)
    if not stats24 or stats24["quote_vol"] < MIN_QUOTE_VOL_24H:
        debug.setdefault("skip", []).append({"symbol": sym, "reason": "low_liquidity_or_24h_fetch"})
        return None
    ob = fetch_orderbook(sym)
    if not ob:
        debug.setdefault("skip", []).append({"symbol": sym, "reason": "no_orderbook"})
        return None
    bid, ask = ob
    spr = spread_pct(bid, ask)
    if spr > MAX_SPREAD:
        debug.setdefault("skip", []).append({"symbol": sym, "reason": f"spread>{MAX_SPREAD:.3f}"})
        return None

    # Pull klines
    k5 = fetch_klines(sym, "5m", 240)    # ~20 hours
    k1h = fetch_klines(sym, "1h", 300)   # ~12.5 days
    if k5 is None or k1h is None or len(k5) < (ATR_LEN_5M + BREAKOUT_LOOKBACK_5M + 5) or len(k1h) < (HTF_CONFIRM_EMA + 5):
        debug.setdefault("skip", []).append({"symbol": sym, "reason": "insufficient_bars"})
        return None

    # Indicators
    k5 = k5.rename(columns={"t":"t","o":"o","h":"h","l":"l","c":"c","v":"v"})
    k1h = k1h.rename(columns={"t":"t","o":"o","h":"h","l":"l","c":"c","v":"v"})
    k5["ema"] = ema(k5["c"], EMA_LEN)
    k5["atr"] = atr(k5, ATR_LEN_5M)
    k5["vwap"] = vwap(k5)
    k1h["ema"] = ema(k1h["c"], HTF_CONFIRM_EMA)
    k1h["vwap"] = vwap(k1h)

    # Last closed bars
    last5 = k5.index[-1]
    prev5 = k5.index[-2]
    last1h = k1h.index[-1]

    # Breakout on 5m: close > prior 20-bar high; rvol>=3; +1.2..6% bar
    hh = float(k5["h"].iloc[-BREAKOUT_LOOKBACK_5M-1:-1].max())
    close = float(k5.at[last5, "c"])
    prev_close = float(k5.at[prev5, "c"])
    bar_ret = (close / prev_close) - 1.0
    rvol = rel_volume(k5, RVOL_WINDOW_5M)

    if not (close > hh and RVOL_MIN <= rvol and BAR_RET_MIN <= bar_ret <= BAR_RET_MAX):
        debug.setdefault("skip", []).append({"symbol": sym, "reason": "no_5m_breakout", "rvol": rvol, "bar_ret": bar_ret})
        return None

    # 1h confirmation: trend above EMA and VWAP
    if HTF_CONFIRM:
        h_ok = (k1h["c"].iloc[-1] > k1h["ema"].iloc[-1]) and (k1h["c"].iloc[-1] > k1h["vwap"].iloc[-1])
        if not h_ok:
            debug.setdefault("skip", []).append({"symbol": sym, "reason": "1h_not_confirmed"})
            return None

    # Stop = 5m breakout bar low; targets from ATR(5m)
    atr5 = float(k5.at[last5, "atr"])
    stop = float(k5.at[last5, "l"])
    entry = close
    t1 = entry + 0.8 * atr5
    t2 = entry + 1.5 * atr5

    return {
        "ticker": sym,
        "entry": entry,
        "stop": stop,
        "atr5m": atr5,
        "t1": t1,
        "t2": t2,
        "rvol5m": rvol,
        "bar_ret": bar_ret,
        "spread": spr,
        "quote_vol_24h": stats24["quote_vol"]
    }

def btc_regime() -> Dict[str, Any]:
    # BTC 5m VWAP/EMA posture for short-term trending environment
    k = fetch_klines("BTC", "5m", 288)
    if k is None or len(k) < 60:
        return {"ok": False, "reason": "no_btc_5m"}
    k = k.rename(columns={"t":"t","o":"o","h":"h","l":"l","c":"c","v":"v"})
    e9 = ema(k["c"], 9).iloc[-1]
    vw = vwap(k).iloc[-1]
    last = k["c"].iloc[-1]
    ok = (last > e9) and (last > vw)
    return {"ok": bool(ok), "last": float(last), "ema9": float(e9), "vwap": float(vw)}

# ---------------- Main
def main():
    t0 = time.time()
    logs: Dict[str, Any] = {"start": now_iso(), "events": []}

    # Floor/guard
    if (EQUITY - EQUITY_FLOOR) < 1000.0 or EQUITY <= (EQUITY_FLOOR + 500.0):
        # hard raise cash signal
        signals = {"type": "A", "text": "Raise cash now: halt entries; sell weakest on breaks."}
        summary = {
            "status": "ok",
            "ts_utc": now_iso(),
            "regime": {"ok": False, "reason": "equity_floor_guard"},
            "equity": EQUITY, "cash": CASH,
            "candidates": [],
            "signals": signals
        }
        SUMMARY.write_text(json.dumps(summary, indent=2))
        SIGNALS.write_text(json.dumps(signals, indent=2))
        RUNSTATS.write_text(json.dumps({"duration_sec": round(time.time()-t0,2), "wrote": "guard"}, indent=2))
        SNAPSHOT.write_text(json.dumps({"note":"guard_triggered"}, indent=2))
        DEBUG.write_text(json.dumps(logs, indent=2))
        print("[analyses] guard fired; wrote raise-cash signal")
        return

    universe = get_mapping_universe()
    logs["events"].append({"universe_count": len(universe)})
    log("universe size:", len(universe))

    regime = btc_regime()
    strong = bool(regime.get("ok", False))
    logs["events"].append({"regime": regime})

    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for i, sym in enumerate(universe, 1):
        if sym in UNTOUCHABLE:
            skipped.append({"symbol": sym, "reason": "untouchable"})
            continue
        # Revolut universe includes fiat or odd tickers sometimes; require Binance spot
        probe = http_get_json(TICKER24, {"symbol": f"{sym}USDT"})
        if probe is None or isinstance(probe, list) or "lastPrice" not in probe:
            skipped.append({"symbol": sym, "reason": "no_binance_spot"})
            continue
        try:
            c = candidate_for_symbol(sym, logs)
            if c:
                candidates.append(c)
        except Exception as e:
            skipped.append({"symbol": sym, "reason": f"exception:{type(e).__name__}"})

        if i % 25 == 0:
            log(f"scanned {i}/{len(universe)}")

    # Rank candidates by quality: higher RVOL, lower spread, higher 24h quote vol, moderate bar_ret
    def score(x: Dict[str, Any]) -> float:
        r = x.get("rvol5m", 0.0)
        spr = x.get("spread", 1.0)
        qv = math.log10(max(1.0, x.get("quote_vol_24h", 1.0)))
        br = x.get("bar_ret", 0.0)
        # prefer bar_ret ~2-4%
        br_bonus = -abs(br - 0.03) * 10
        return 3.0*r + 1.5*qv + br_bonus - 5.0*spr

    candidates.sort(key=score, reverse=True)

    # Build signals
    if candidates and strong:
        top = candidates[0]
        entry = float(top["entry"]); stop = float(top["stop"])
        buy_usd = position_usd(entry, stop, EQUITY, CASH, strong)
        plan = {
            "type": "B",
            "ticker": top["ticker"],
            "entry": round(entry, 6),
            "stop": round(stop, 6),
            "T1": round(top["t1"], 6),
            "T2": round(top["t2"], 6),
            "atr5m": round(top["atr5m"], 6),
            "rvol5m": round(top["rvol5m"], 3),
            "bar_ret": round(top["bar_ret"], 4),
            "spread": round(top["spread"], 5),
            "position_usd": round(buy_usd, 2),
            "note": "trail after +1R; sell weakest on breaks; do not rotate ETH/DOT; DOT is staked."
        }
        signals = plan
    elif candidates:
        # No strong regime; still surface top candidates for review
        view = []
        for x in candidates[:5]:
            view.append({
                "ticker": x["ticker"],
                "entry": round(float(x["entry"]),6),
                "stop": round(float(x["stop"]),6),
                "T1": round(float(x["t1"]),6),
                "T2": round(float(x["t2"]),6),
                "rvol5m": round(float(x["rvol5m"]),3),
                "bar_ret": round(float(x["bar_ret"]),4),
                "spread": round(float(x["spread"]),5)
            })
        signals = {"type": "C", "text": "Hold and wait; regime weak.", "top": view}
    else:
        signals = {"type": "C", "text": "Hold and wait. No qualified candidates."}

    # Write outputs
    summary = {
        "status": "ok",
        "ts_utc": now_iso(),
        "equity": EQUITY,
        "cash": CASH,
        "regime": regime,
        "candidates": candidates[:10],
        "signals": signals
    }
    snapshot = {
        "universe_count": len(universe),
        "scanned": len(universe) - len(UNTOUCHABLE),
        "skipped_count": len(skipped),
        "skipped": skipped[:100]
    }
    runstats = {
        "duration_sec": round(time.time()-t0, 2),
        "time_utc": now_iso()
    }

    SUMMARY.write_text(json.dumps(summary, indent=2))
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
    DEBUG.write_text(json.dumps({"log": "see events", "events": logs["events"]}, indent=2))
    RUNSTATS.write_text(json.dumps(runstats, indent=2))
    SIGNALS.write_text(json.dumps(signals, indent=2))

    print(f"[analyses] wrote summary with {len(candidates)} candidates in {runstats['duration_sec']}s", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Always emit minimal outputs on fatal error
        fail = {
            "status": "error",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(e).__name__}: {e}",
            "signals": {"type":"C","text":"Hold and wait. Exception."}
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(fail, indent=2))
        SNAPSHOT.write_text(json.dumps({"fatal": True}, indent=2))
        DEBUG.write_text(json.dumps({"traceback": traceback.format_exc()}, indent=2))
        RUNSTATS.write_text(json.dumps({"duration_sec": None, "time_utc": now_iso(), "fatal": True}, indent=2))
        print("[analyses] FAILED but wrote minimal outputs", flush=True)
        raise