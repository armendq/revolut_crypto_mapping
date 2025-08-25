#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyses.py
-----------
Scans a broad crypto universe (from your mapping file), fetches hourly data
from Binance, detects fresh breakouts, computes ATR and targets, then writes:
  public_runs/latest/summary.json
  public_runs/latest/run_stats.json
  public_runs/latest/market_snapshot.json
  public_runs/latest/debug_scan.json

Design goals:
- Wider universe (no hard-coded short list).
- Transparent filtering (+ counts).
- Robust, no exceptions bubbling up (graceful skips).
- Signals:
    "B"  -> fresh breakouts detected on the latest closed bar
    "C"  -> no fresh breakouts; show top candidates (momentum + volume surge)
- Rules respected:
    * DOT is staked/untouchable.
    * ETH and DOT are never “rotated out” by this script.
    * Stop = breakout bar LOW.
    * Targets: T1 = entry + 0.8*ATR, T2 = entry + 1.5*ATR.
- Regime: simple BTC hourly regime check.

Requires internet access in the runner (Binance public REST).
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import traceback
from typing import List, Dict, Any, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# --------------- Configuration ---------------

# Where to write results
OUT_DIR = os.path.join("public_runs", "latest")

# Mapping file(s) to discover symbols
MAPPING_CANDIDATES = [
    os.path.join("data", "mapping.json"),
    os.path.join("mapping.json"),
    os.path.join("data", "mapping_all.json"),
]

# Binance REST
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"  # ?symbol=BTCUSDT&interval=1h&limit=500

# Universe & filters
INTERVAL = "1h"
LIMIT = 500                    # last 500 hourly candles
LOOKBACK_BREAKOUT = 20         # highest-high lookback (ex-current) for breakout
ATR_LEN = 14
MIN_QUOTE_VOL_24H = 1_000_000  # rough liquidity floor via 24h quote vol from klines sum
VOLUME_SPIKE_MULT = 1.5        # latest bar volume vs avg(20)
MAX_CANDIDATES = 15            # when in "C" mode

# Sizing hints (these are only hints; final sizing happens downstream)
RISK_PCT = 0.012               # 1.2%

# Do not rotate / trade rules
DO_NOT_ROTATE = {"ETH", "DOT"}
UNTOUCHABLE = {"DOT"}          # staked

# --------------- Utilities ---------------

def ensure_outdir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def try_load_mapping() -> List[Dict[str, Any]]:
    for p in MAPPING_CANDIDATES:
        if os.path.exists(p):
            data = read_json(p)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "assets" in data and isinstance(data["assets"], list):
                return data["assets"]
    # Fallback minimal universe if mapping missing
    return [
        {"symbol": "BTC", "binance": "BTCUSDT"},
        {"symbol": "ETH", "binance": "ETHUSDT"},
        {"symbol": "SOL", "binance": "SOLUSDT"},
        {"symbol": "ARB", "binance": "ARBUSDT"},
        {"symbol": "IMX", "binance": "IMXUSDT"},
        {"symbol": "OP",  "binance": "OPUSDT"},
        {"symbol": "DOT", "binance": "DOTUSDT"},
    ]

def http_get_json(url: str) -> Optional[Any]:
    req = Request(url, headers={"User-Agent": "gh-actions-analysis/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except (URLError, HTTPError, json.JSONDecodeError):
        return None

def fetch_klines_binance(symbol: str, interval: str, limit: int) -> Optional[List[List[Any]]]:
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    data = http_get_json(url)
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data
    return None

def array_high(arr: List[float]) -> float:
    return max(arr) if arr else float("nan")

def array_low(arr: List[float]) -> float:
    return min(arr) if arr else float("nan")

def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def sma(values: List[float], length: int) -> List[float]:
    out: List[float] = []
    window_sum = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        window_sum += v
        if len(q) > length:
            window_sum -= q.pop(0)
        out.append(window_sum / len(q))
    return out

def atr(highs: List[float], lows: List[float], closes: List[float], length: int) -> List[float]:
    trs: List[float] = []
    for i in range(len(closes)):
        if i == 0:
            trs.append(highs[i] - lows[i])
        else:
            trs.append(true_range(highs[i], lows[i], closes[i - 1]))
    return sma(trs, length)

def sum_last(values: List[float], n: int) -> float:
    return float(sum(values[-n:])) if values else 0.0

# --------------- Core analysis ---------------

def analyze_one(symbol: str, binance: str) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with computed metrics (even if no breakout),
    or None if we couldn't fetch/process data.
    """
    kl = fetch_klines_binance(binance, INTERVAL, LIMIT)
    if not kl or len(kl) < LOOKBACK_BREAKOUT + 2:
        return None

    # Parse arrays
    # Binance kline: [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, trades, ... ]
    highs = [float(k[2]) for k in kl]
    lows  = [float(k[3]) for k in kl]
    closes = [float(k[4]) for k in kl]
    vols = [float(k[5]) for k in kl]
    quote_vols = [float(k[7]) for k in kl]

    # 24h liquidity proxy (24 candles if 1h)
    liq_24h = sum_last(quote_vols, 24)

    # Basic guards
    if liq_24h < MIN_QUOTE_VOL_24H:
        return {
            "symbol": symbol,
            "binance": binance,
            "skip": "low_liquidity",
            "liq_24h": liq_24h
        }

    # Indicators
    atr_arr = atr(highs, lows, closes, ATR_LEN)
    atr_val = atr_arr[-2]  # latest CLOSED bar ATR
    if atr_val <= 0 or math.isnan(atr_val):
        return None

    # Latest CLOSED bar = index -2 (last element is still-forming)
    i = len(closes) - 2
    last_close = closes[i]
    last_high = highs[i]
    last_low  = lows[i]
    last_vol  = vols[i]

    # Breakout setup: compare last close vs highest-high of previous LOOKBACK bars (exclude current bar)
    prev_window_high = array_high(highs[i - LOOKBACK_BREAKOUT : i]) if i - LOOKBACK_BREAKOUT >= 0 else array_high(highs[:i])
    vol_avg20 = sum_last(vols[:i], 20) / 20.0 if i >= 20 else sum_last(vols[:i], max(1, i)) / max(1, min(i, 20))
    vol_spike = (last_vol / vol_avg20) if vol_avg20 > 0 else 0.0

    is_breakout = (last_close > prev_window_high) and (vol_spike >= VOLUME_SPIKE_MULT)

    entry = max(last_close, prev_window_high) if is_breakout else prev_window_high
    stop = last_low  # breakout bar low (rule)

    t1 = entry + 0.8 * atr_val
    t2 = entry + 1.5 * atr_val

    # Simple momentum score: distance from SMA20 + volume surge
    smas = sma(closes, 20)
    mom = (last_close / smas[i] - 1.0) if smas[i] > 0 else 0.0
    score = 0.7 * mom + 0.3 * min(vol_spike / 3.0, 1.0)

    return {
        "symbol": symbol,
        "binance": binance,
        "liq_24h": liq_24h,
        "atr": atr_val,
        "prev_window_high": prev_window_high,
        "last_close": last_close,
        "last_low": last_low,
        "last_high": last_high,
        "vol_spike": vol_spike,
        "is_breakout": bool(is_breakout),
        "entry": entry,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "score": score
    }

def btc_regime(mapping: List[Dict[str, Any]]) -> Dict[str, Any]:
    btc_symbol = None
    for a in mapping:
        if a.get("symbol") == "BTC":
            btc_symbol = a.get("binance") or "BTCUSDT"
            break
    if not btc_symbol:
        btc_symbol = "BTCUSDT"
    kl = fetch_klines_binance(btc_symbol, INTERVAL, LIMIT)
    if not kl or len(kl) < 210:
        return {"ok": False, "reason": "insufficient_bars"}
    closes = [float(k[4]) for k in kl]
    sma200 = sma(closes, 200)
    ok = closes[-2] > sma200[-2]  # last closed bar vs SMA200
    return {
        "ok": bool(ok),
        "trend_ref": "SMA200",
        "price": closes[-2],
        "sma200": sma200[-2]
    }

def main() -> None:
    start_ts = int(time.time())
    ensure_outdir()

    mapping = try_load_mapping()

    # Build universe from mapping (prefer entries that define a Binance symbol)
    universe: List[Dict[str, str]] = []
    for a in mapping:
        sym = str(a.get("symbol", "")).upper()
        binance = a.get("binance")
        if not sym or not binance:
            continue
        # Avoid obvious duplicates
        if not any(x["binance"] == binance for x in universe):
            universe.append({"symbol": sym, "binance": binance})

    # If mapping was empty, universe contains a sane fallback already
    universe_count = len(universe)

    results: List[Dict[str, Any]] = []
    skipped = {"low_liquidity": 0, "error": 0, "other": 0}

    for u in universe:
        try:
            r = analyze_one(u["symbol"], u["binance"])
            if r is None:
                skipped["other"] += 1
                continue
            # Always collect (even skipped w/ reason) for debug visibility
            results.append(r)
        except Exception:
            skipped["error"] += 1
            results.append({
                "symbol": u["symbol"],
                "binance": u["binance"],
                "skip": "exception",
                "error": traceback.format_exc(limit=1)
            })

    # Separate valid metrics
    valid = [x for x in results if "atr" in x and "entry" in x]

    # Fresh breakouts for signal "B"
    breakouts = [x for x in valid if x.get("is_breakout") and x["symbol"] not in UNTOUCHABLE]

    # Candidate ranking (no fresh breakout -> "C")
    candidates_pool = [x for x in valid if x["symbol"] not in UNTOUCHABLE]
    candidates_pool.sort(key=lambda z: z.get("score", 0.0), reverse=True)
    candidates = candidates_pool[:MAX_CANDIDATES]

    # Regime
    regime_info = btc_regime(mapping)

    # Prepare outputs
    market_snapshot = {
        "time_unix": start_ts,
        "universe_count": universe_count,
        "scanned": len(results),
        "valid": len(valid),
        "skipped": skipped
    }

    swing = {
        "breakouts": breakouts
    }

    fast = {
        "candidates": candidates
    }

    # Signal selection
    if breakouts:
        signal = {"type": "B", "count": len(breakouts)}
    else:
        signal = {"type": "C", "count": len(candidates)}

    summary = {
        "status": "ok",
        "time_unix": start_ts,
        "regime": regime_info,
        "signals": signal,
        "notes": {
            "risk_pct_hint": RISK_PCT,
            "do_not_rotate": sorted(list(DO_NOT_ROTATE)),
            "untouchable": sorted(list(UNTOUCHABLE)),
            "interval": INTERVAL,
            "lookback_breakout": LOOKBACK_BREAKOUT,
            "atr_len": ATR_LEN,
            "volume_spike_mult": VOLUME_SPIKE_MULT,
            "min_quote_vol_24h": MIN_QUOTE_VOL_24H
        }
    }

    # Persist files
    try:
        with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(OUT_DIR, "run_stats.json"), "w", encoding="utf-8") as f:
            json.dump(market_snapshot, f, indent=2)
        with open(os.path.join(OUT_DIR, "market_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump({"swing": swing, "fast": fast}, f, indent=2)
        # Full debug dump (so we can inspect why assets got filtered out)
        with open(os.path.join(OUT_DIR, "debug_scan.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception:
        # As a last resort, print to stdout to not fail the job
        print("[warn] Failed to write outputs:", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()