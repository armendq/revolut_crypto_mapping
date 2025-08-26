#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import math
import pathlib
import statistics
from typing import Dict, Any, List, Tuple, Optional
import requests

# ---------------------------
# Paths
# ---------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PUB_DIR = ROOT / "public_runs" / "latest"
PUB_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = PUB_DIR / "summary.json"
DEBUG_PATH = PUB_DIR / "debug.json"
LOG_PATH = PUB_DIR / "run.log"
MAPPING_FILE = DATA_DIR / "revolut_mapping.json"

# ---------------------------
# Config
# ---------------------------

# Use the only working Binance mirror per your note
BINANCE_BASE = "https://data-api.binance.vision"
KLINES_EP = "/api/v3/klines"        # requires: symbol, interval, limit
TICKER24_EP = "/api/v3/ticker/24hr" # optional metadata (spread/vol), kept for future use

# Headers to reduce 451/problems
REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; rev-analyses/1.0; +https://github.com/armendq/revolut_crypto_mapping)",
    "Accept": "application/json,text/plain,*/*",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Origin": "https://data-api.binance.vision",
    "Referer": "https://data-api.binance.vision/",
}

TIMEOUT = 20
RETRIES = 4
BACKOFF = 1.6

# Spike detection parameters (tuneable)
INTERVAL = "1h"      # 1-hour bars
BARS = 120           # history depth
WINDOW_H = 8         # lookback window for burst
MIN_PCT_RISE = 0.20  # >= +20% over last 8h
VOL_MULT = 1.8       # last-8h volume vs prior-8h
ATR_LEN = 14

# Pace to avoid rate limits
SLEEP_BETWEEN = 0.06
CHUNK_PAUSE = 1.0
CHUNK_SIZE = 120

# Rotation exclusions
EXCLUDE_ROTATION = {"ETH", "DOT"}   # DOT is staked, ETH not rotated

# ---------------------------
# Logging helpers
# ---------------------------

DEBUG_LOG: Dict[str, Any] = {"events": []}

def log(event: str, **fields):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    rec = {"t": ts, "event": event}
    rec.update(fields)
    DEBUG_LOG["events"].append(rec)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"[analyses] {ts} {event} {fields}\n")

def write_json(path: pathlib.Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ---------------------------
# HTTP helpers
# ---------------------------

def get_json(url: str, params: Dict[str, Any]) -> Any:
    last_err: Optional[str] = None
    for i in range(1, RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=REQ_HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_err = f"http {r.status_code}"
            log("http_non_200", url=url, status=r.status_code, try_num=i)
        except Exception as e:
            last_err = str(e)
            log("http_exc", url=url, err=last_err, try_num=i)
        time.sleep(BACKOFF ** (i - 1))
    raise RuntimeError(f"GET failed: {url} params={params} err={last_err}")

def fetch_klines(symbol: str, interval: str = INTERVAL, limit: int = BARS) -> List[List[Any]]:
    url = BINANCE_BASE + KLINES_EP
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = get_json(url, params)
    if not isinstance(data, list) or not data:
        raise RuntimeError("Empty klines")
    return data

# ---------------------------
# Mapping
# ---------------------------

def load_mapping() -> List[Dict[str, str]]:
    """
    Accepts:
      - list of dicts with keys {'revolut_ticker','binance_symbol'}
      - dict of {revolut_ticker: binance_symbol}
      - dict of {revolut_ticker: anything} -> fallback to f"{revolut_ticker}USDT"
      - list of strings -> fallback to f"{item}USDT"
    Returns list[{revolut_ticker, binance_symbol}]
    """
    if not MAPPING_FILE.exists():
        raise FileNotFoundError("data/revolut_mapping.json not found.")

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: List[Dict[str, str]] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                rev = str(item.get("revolut_ticker") or item.get("ticker") or "").upper().strip()
                sym = str(item.get("binance_symbol") or "").upper().strip()
                if not rev:
                    continue
                if not sym:
                    sym = f"{rev}USDT"
                out.append({"revolut_ticker": rev, "binance_symbol": sym})
            elif isinstance(item, str):
                rev = item.upper().strip()
                if rev:
                    out.append({"revolut_ticker": rev, "binance_symbol": f"{rev}USDT"})
    elif isinstance(raw, dict):
        for rev, val in raw.items():
            rev_u = str(rev).upper().strip()
            sym = ""
            if isinstance(val, str):
                # if looks like a Binance symbol use that, else fallback to USDT pair
                val_u = val.upper().strip()
                sym = val_u if val_u.endswith("USDT") else f"{rev_u}USDT"
            else:
                sym = f"{rev_u}USDT"
            out.append({"revolut_ticker": rev_u, "binance_symbol": sym})
    else:
        raise ValueError("Unsupported mapping JSON schema.")

    # Deduplicate by revolut ticker
    dedup: Dict[str, Dict[str, str]] = {}
    for row in out:
        rev = row["revolut_ticker"]
        sym = row["binance_symbol"]
        if rev and sym:
            dedup[rev] = {"revolut_ticker": rev, "binance_symbol": sym}

    final = list(dedup.values())
    log("mapping_loaded", count=len(final))
    return final

# ---------------------------
# Math helpers
# ---------------------------

def f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan

def calc_atr(kl: List[List[Any]], length: int = ATR_LEN) -> float:
    trs: List[float] = []
    for i in range(1, len(kl)):
        pc = f(kl[i-1][4])
        hi = f(kl[i][2])
        lo = f(kl[i][3])
        tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
        trs.append(tr)
    if len(trs) < length:
        return float("nan")
    return statistics.fmean(trs[-length:])

def detect_spike(kl: List[List[Any]]) -> Dict[str, Any]:
    """
    Check last 8h momentum and volume burst.
    Returns {} if no signal, else {entry, stop, atr, t1, t2, pct8h, vol_mult}.
    """
    need = ATR_LEN + WINDOW_H + 5
    if len(kl) < need:
        return {}

    closes = [f(r[4]) for r in kl]
    lows   = [f(r[3]) for r in kl]
    vols   = [f(r[5]) for r in kl]

    last_close = closes[-1]
    ref_close = closes[-(WINDOW_H + 1)]
    if not (math.isfinite(last_close) and math.isfinite(ref_close)) or ref_close <= 0:
        return {}

    pct = (last_close - ref_close) / ref_close
    last8_vol = sum(vols[-WINDOW_H:])
    prev8_vol = sum(vols[-(2*WINDOW_H):-WINDOW_H]) + 1e-12
    vol_mult = last8_vol / prev8_vol

    if pct < MIN_PCT_RISE or vol_mult < VOL_MULT:
        return {}

    atr = calc_atr(kl, ATR_LEN)
    if not math.isfinite(atr) or atr <= 0:
        return {}

    # Conservative stop: min low in the window
    stop = min(lows[-WINDOW_H:])
    entry = last_close
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr

    return {
        "entry": round(entry, 8),
        "stop": round(stop, 8),
        "atr": round(atr, 8),
        "t1": round(t1, 8),
        "t2": round(t2, 8),
        "pct8h": round(pct * 100.0, 2),
        "vol_mult": round(vol_mult, 2),
    }

# ---------------------------
# Main
# ---------------------------

def main() -> int:
    t0 = time.time()
    write_json(DEBUG_PATH, {"events": []})  # reset debug

    try:
        mapping = load_mapping()
    except Exception as e:
        log("fatal_mapping", err=str(e))
        write_json(SUMMARY_PATH, {
            "status": "error",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": f"mapping: {e}",
            "signals": {"type": "C", "text": "Hold and wait."},
            "candidates": []
        })
        write_json(DEBUG_PATH, DEBUG_LOG)
        return 1

    # Prepare universe, exclude non-rotated tickers
    universe: List[Tuple[str, str]] = []
    for row in mapping:
        rev = row["revolut_ticker"].upper().strip()
        sym = row["binance_symbol"].upper().strip()
        if not rev or not sym:
            continue
        if rev in EXCLUDE_ROTATION:   # exclude ETH and DOT from signals
            continue
        # Simple sanity: symbol must end with USDT to be spot pair
        if not sym.endswith("USDT"):
            continue
        universe.append((rev, sym))

    total = len(universe)
    log("universe", count=total)

    candidates: List[Dict[str, Any]] = []
    scanned = 0

    for idx, (rev, sym) in enumerate(universe, 1):
        try:
            kl = fetch_klines(sym, interval=INTERVAL, limit=BARS)  # WITH symbol parameter
            sig = detect_spike(kl)
            if sig:
                candidates.append({
                    "revolut_ticker": rev,
                    "binance_symbol": sym,
                    **sig
                })
        except Exception as e:
            log("symbol_fail", sym=sym, err=str(e))
        finally:
            scanned += 1
            if scanned % 25 == 0 or scanned == total:
                log("progress", scanned=scanned, total=total)
            time.sleep(SLEEP_BETWEEN)
        if idx % CHUNK_SIZE == 0:
            time.sleep(CHUNK_PAUSE)

    # Sort strongest first
    candidates.sort(key=lambda x: (x.get("pct8h", 0.0), x.get("vol_mult", 0.0)), reverse=True)

    # Build summary
    summary: Dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "universe": total,
        "scanned": scanned,
        "params": {
            "interval": INTERVAL,
            "window_hours": WINDOW_H,
            "min_pct_rise": MIN_PCT_RISE,
            "vol_mult": VOL_MULT,
            "atr_len": ATR_LEN
        },
        "signals": {},
        "candidates": candidates
    }

    if candidates:
        summary["signals"] = {"type": "C", "text": "Candidates available."}
    else:
        summary["signals"] = {"type": "C", "text": "Hold and wait."}

    write_json(SUMMARY_PATH, summary)
    write_json(DEBUG_PATH, DEBUG_LOG)
    log("done", took_s=round(time.time() - t0, 2), candidates=len(candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())