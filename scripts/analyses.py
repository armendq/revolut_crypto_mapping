#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses runner
- Loads Revolut↔Binance mapping (expects keys: binance_symbol, revolut_ticker).
- Scans each mapped Binance symbol with 15m klines (~8h window).
- Computes 8h momentum, ATR(14), and simple quality filters.
- Writes concise summary to public_runs/latest/summary.json.

This version fixes the root cause of prior 400/451 errors by ALWAYS
supplying the required `symbol` parameter to Binance kline endpoints.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import requests

# ----------------------------
# Config
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
PUBLIC_DIR = os.path.join(ROOT, "public_runs", "latest")
os.makedirs(PUBLIC_DIR, exist_ok=True)

# Where to read mapping from (generated earlier in the workflow)
# Must be a JSON array of objects: [{"binance_symbol":"BTCUSDT","revolut_ticker":"BTC"}, ...]
DEFAULT_MAPPING_FILES = [
    os.path.join(DATA_DIR, "revolut_mapping.json"),           # preferred output of generate_mapping.py
    os.path.join(DATA_DIR, "mapping.json"),                   # fallback name if used
]

USER_AGENT = "revolut-analyses/1.0 (+https://github.com/armendq/revolut_crypto_mapping)"
BINANCE_BASE = "https://api.binance.com"
BINANCE_VISION = "https://data-api.binance.vision"

# Scan parameters
INTERVAL = "15m"     # 15-min candles
LIMIT = 64           # ~16 hours; we'll use last 32 (~8h) for momentum calc
RATE_SLEEP = 0.20    # 5 req/sec conservative
MAX_RETRIES = 4
TIMEOUT = 10

# Candidate rules
MIN_8H_MOVE_PCT = 18.0        # >= 18% move over ~8h
MIN_USD_VOLUME_8H = 500_000   # dollar volume over ~8h
MAX_SPREAD_PCT = 1.5          # simple spread proxy using (high/low - 1) on last candle
EXCLUDE_SYMBOLS = set()       # optionally add exclusions

# ----------------------------
# Utility
# ----------------------------

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[analyses] {ts} UTC {msg}", flush=True)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, payload: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    os.replace(tmp, path)

def ensure_mapping_schema(obj: Any) -> List[Dict[str, str]]:
    """
    Accept either:
      - List[dict] with keys {'binance_symbol','revolut_ticker'}
      - Dict[str,str] like {'BTC':'Bitcoin', ...}  -> cannot use (no binance_symbol), reject
    """
    if isinstance(obj, list):
        ok = []
        for row in obj:
            if not isinstance(row, dict):
                continue
            b = row.get("binance_symbol")
            r = row.get("revolut_ticker")
            if b and r:
                ok.append({"binance_symbol": b.upper(), "revolut_ticker": r.upper()})
        if ok:
            return ok
    # Any other shape is not acceptable for scanning
    raise FileNotFoundError(
        "[analyses] No mapping file found with expected keys "
        "['binance_symbol','revolut_ticker']."
    )

def load_mapping() -> List[Dict[str, str]]:
    for path in DEFAULT_MAPPING_FILES:
        if os.path.exists(path):
            try:
                raw = read_json(path)
                mapping = ensure_mapping_schema(raw)
                log(f"loaded mapping from {os.path.relpath(path, ROOT)} with {len(mapping)} rows")
                return mapping
            except Exception as e:
                log(f"{os.path.relpath(path, ROOT)} found but schema not matching; skipping.")
    raise FileNotFoundError(
        "[analyses] No mapping file found with expected keys "
        "['binance_symbol','revolut_ticker']. Add mapping or adjust DEFAULT_MAPPING_FILES."
    )

# ----------------------------
# Binance client (public)
# ----------------------------

class Http:
    def __init__(self, base: str):
        self.base = base
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        url = self.base + path
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = self.s.get(url, params=params, timeout=TIMEOUT)
                if r.status_code == 200:
                    return r
                else:
                    log(f"GET fail: {r.url} http {r.status_code} (try {attempt})")
            except Exception as e:
                log(f"GET error: {url} ({e}) (try {attempt})")
            time.sleep(min(2 ** (attempt - 1), 4))
        r.raise_for_status()
        return r  # unreachable

binance_api = Http(BINANCE_BASE)

def fetch_klines(symbol: str, interval: str = INTERVAL, limit: int = LIMIT) -> List[List[Any]]:
    """
    Returns raw klines:
    [ openTime, open, high, low, close, volume, closeTime, qav, trades, takerBase, takerQuote, ignore ]
    """
    # CRITICAL: include symbol in params (prevents 400/451)
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = binance_api.get("/api/v3/klines", params=params)
    time.sleep(RATE_SLEEP)
    return r.json()

# ----------------------------
# Math helpers
# ----------------------------

def to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def pct_change(a: float, b: float) -> float:
    if a == 0 or math.isnan(a) or math.isnan(b):
        return 0.0
    return (b - a) / a * 100.0

def atr_from_klines(rows: List[List[Any]], n: int = 14) -> float:
    """
    ATR(14) using classic Wilder's (simple rolling TR avg here is sufficient).
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    if len(rows) < n + 1:
        return float("nan")
    trs: List[float] = []
    prev_close = to_float(rows[0][4])
    for r in rows[1:]:
        high = to_float(r[2])
        low = to_float(r[3])
        close = to_float(r[4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        prev_close = close
    if not trs:
        return float("nan")
    # average of last n TR
    window = trs[-n:]
    return sum(window) / float(len(window))

def dollar_volume_approx(rows: List[List[Any]]) -> float:
    """
    Approximate dollar volume over the klines set:
      sum(close * volume) using base asset volume * close price in quote (USDT)
    """
    s = 0.0
    for r in rows:
        close = to_float(r[4])
        vol_base = to_float(r[5])
        s += close * vol_base
    return s

# ----------------------------
# Regime (very light)
# ----------------------------

def btc_regime_ok() -> bool:
    try:
        rows = fetch_klines("BTCUSDT", interval="1h", limit=200)
        closes = [to_float(r[4]) for r in rows]
        if len(closes) < 50:
            return True
        ema = closes[0]
        alpha = 2 / (50 + 1)
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return closes[-1] >= ema
    except Exception as e:
        log(f"regime check failed: {e}")
        return True  # default permissive

# ----------------------------
# Candidate selection
# ----------------------------

@dataclass
class Candidate:
    binance_symbol: str
    revolut_ticker: str
    last: float
    atr: float
    move8h_pct: float
    usd_vol_8h: float
    spread_proxy_pct: float

    def to_dict(self) -> Dict[str, Any]:
        entry = self.last
        atr = self.atr
        return {
            "symbol": self.binance_symbol,
            "ticker": self.revolut_ticker,
            "last": round(entry, 10),
            "atr": round(atr, 10),
            "entry": round(entry, 10),
            # If you want true breakout-bar-low, you can swap in that value here.
            "stop": round(max(0.0, entry - atr), 10),
            "T1": round(entry + 0.8 * atr, 10),
            "T2": round(entry + 1.5 * atr, 10),
            "move8h_pct": round(self.move8h_pct, 3),
            "usd_vol_8h": round(self.usd_vol_8h, 2),
            "spread_proxy_pct": round(self.spread_proxy_pct, 3),
        }

def evaluate_symbol(binance_symbol: str, revolut_ticker: str) -> Optional[Candidate]:
    if binance_symbol in EXCLUDE_SYMBOLS:
        return None
    try:
        rows = fetch_klines(binance_symbol, interval=INTERVAL, limit=LIMIT)
        if not rows or len(rows) < 40:
            return None

        # Use last 32 candles for ~8h window on 15m interval
        last_32 = rows[-32:]
        start = to_float(last_32[0][4])
        end = to_float(last_32[-1][4])
        move = pct_change(start, end)

        # ATR on total fetched set (smoother)
        atr = atr_from_klines(rows, n=14)

        # Spread proxy on last bar
        lh = to_float(rows[-1][2]) / max(to_float(rows[-1][3]), 1e-12) - 1.0
        spread_pct = lh * 100.0

        # Dollar volume approx over last 32 bars
        usd_vol = dollar_volume_approx(last_32)

        # Filters
        if any(math.isnan(x) or x == 0.0 for x in (end, atr)):
            return None
        if move < MIN_8H_MOVE_PCT:
            return None
        if usd_vol < MIN_USD_VOLUME_8H:
            return None
        if spread_pct > MAX_SPREAD_PCT:
            return None

        return Candidate(
            binance_symbol=binance_symbol,
            revolut_ticker=revolut_ticker,
            last=end,
            atr=atr,
            move8h_pct=move,
            usd_vol_8h=usd_vol,
            spread_proxy_pct=spread_pct,
        )
    except Exception as e:
        log(f"{binance_symbol} eval error: {e}")
        return None

# ----------------------------
# Main
# ----------------------------

def main() -> int:
    try:
        mapping = load_mapping()
    except Exception as e:
        log(f"FATAL: {e}")
        sys.exit(1)

    regime_ok = btc_regime_ok()
    log(f"regime.ok={regime_ok}")

    universe = [(m["binance_symbol"].upper(), m["revolut_ticker"].upper()) for m in mapping]
    log(f"universe size: {len(universe)}")

    candidates: List[Candidate] = []
    scanned = 0
    for sym, tick in universe:
        scanned += 1
        if scanned % 25 == 0:
            log(f"scanned {scanned}/{len(universe)}")
        c = evaluate_symbol(sym, tick)
        if c:
            candidates.append(c)

    # Rank by move pct then dollar volume for tie-break
    candidates.sort(key=lambda x: (x.move8h_pct, x.usd_vol_8h), reverse=True)

    # Build summary
    out: Dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": {"ok": bool(regime_ok)},
        "signals": {},
        "candidates": [c.to_dict() for c in candidates],
        "meta": {
            "interval": INTERVAL,
            "limit": LIMIT,
            "min_move_pct_8h": MIN_8H_MOVE_PCT,
            "min_usd_volume_8h": MIN_USD_VOLUME_8H,
            "max_spread_pct": MAX_SPREAD_PCT,
            "scanned": len(universe),
        },
    }

    # If you later promote to "B" (buy) signal, fill out signals accordingly.
    # For now, surface as "C" (candidates) when any exist.
    if candidates:
        out["signals"] = {"type": "C", "note": "Momentum candidates over ~8h"}

    # Write JSON
    out_path = os.path.join(PUBLIC_DIR, "summary.json")
    write_json(out_path, out)
    log(f"wrote summary with {len(candidates)} candidates -> {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())