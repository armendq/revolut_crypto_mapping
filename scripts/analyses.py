#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 analyses.py
 Robust scanner for momentum/breakout candidates with ATR sizing inputs.
 - Discovers mapping file automatically (multiple fallbacks)
 - Rotates Binance endpoints; polite retry/backoff; always sets symbol when needed
 - Scans 15m klines, ~8h lookback (configurable) to catch sustained intraday spikes
 - Builds summary, signals, and debug artifacts in public_runs/latest

 Requirements:
   pip install requests pandas numpy python-dateutil rapidfuzz
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import glob
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone

# ---------------------------- Logging -----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("analyses")


# ---------------------------- Config ------------------------------------------
OUT_DIR = os.environ.get("RUN_OUT_DIR", "public_runs/latest")

# Scan settings
CANDLE_INTERVAL = os.environ.get("SCAN_INTERVAL", "15m")   # 1m/5m/15m/1h
LOOKBACK_HOURS = float(os.environ.get("LOOKBACK_HOURS", "8"))
ATR_LEN = int(os.environ.get("ATR_LEN", "14"))
BREAKOUT_LOOKBACK = int(os.environ.get("BREAKOUT_LOOKBACK", "20"))  # Donchian high
MIN_VOLUME_USDT = float(os.environ.get("MIN_VOLUME_USDT", "150000"))  # 24h filter
MIN_MARKETCAP_USD = float(os.environ.get("MIN_MARKETCAP_USD", "0"))   # if mapping has it
MAX_SYMBOLS = int(os.environ.get("MAX_SYMBOLS", "500"))  # safety

# Sizing defaults (used in downstream consumer; we include in JSON for convenience)
RISK_PCT = float(os.environ.get("RISK_PCT", "0.012"))  # 1.2%

# Rules
DO_NOT_ROTATE = {"ETH", "DOT"}  # non-rotated
UNTOUCHABLE = {"DOT"}  # staked & untouchable for new buys

# Binance endpoints rotation
BINANCE_PUBLIC_BASES = [
    # Official
    "https://api.binance.com",
    # CDN mirror
    "https://data-api.binance.vision",
    # Extra mirrors (kept for resilience)
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

# Headers for nicer treatment
UA = {"User-Agent": "revolut-crypto-mapper/analyses (+github actions; contact owner)"}

SESSION = requests.Session()
SESSION.headers.update(UA)
SESSION_TIMEOUT = (7, 20)  # connect, read


# --------------------- Utilities: time & json safe ----------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def json_dump(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


# -------------------------- Mapping discovery ---------------------------------
EXPECTED_KEYS = {"revolut_ticker", "binance_symbol"}

DEFAULT_MAPPING_FILES = [
    "mapping/generated_mapping.json",
    "data/revolut_mapping.json",
    "data/mapping.json",
    "mapping.json",
    "public/mapping.json",
    "public_runs/latest/mapping.json",
    "public_runs/mapping.json",
    "outputs/mapping.json",
]


def _looks_like_mapping(obj: Any) -> bool:
    if isinstance(obj, list) and obj:
        sample = obj[0]
        return isinstance(sample, dict) and EXPECTED_KEYS.issubset(sample.keys())
    return False


def load_mapping() -> List[Dict[str, Any]]:
    for p in DEFAULT_MAPPING_FILES:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if _looks_like_mapping(data):
                    log.info(f"[analyses] using mapping: {p} ({len(data)} records)")
                    return data
                else:
                    log.info(f"[analyses] {p} found but schema not matching; skipping.")
            except Exception as e:
                log.info(f"[analyses] failed to read {p}: {e}")

    # Fallback: scan repo for any plausible mapping JSON
    candidates: List[Tuple[str, int]] = []
    for path in glob.glob("**/*.json", recursive=True):
        if any(seg in path.lower() for seg in ["node_modules", ".git", ".venv", "site-packages", "public_runs"]):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read(300_000)
                obj = json.loads(txt)
            if _looks_like_mapping(obj):
                candidates.append((path, len(obj)))
        except Exception:
            pass
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen, n = candidates[0]
        log.info(f"[analyses] discovered mapping: {chosen} ({n} records)")
        with open(chosen, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(
        "[analyses] No mapping file found with expected keys "
        f"{sorted(EXPECTED_KEYS)}. Add mapping or adjust DEFAULT_MAPPING_FILES."
    )


# ----------------------- Binance API helpers ----------------------------------
def _get_with_rotation(path: str, params: Dict[str, Any] | None = None,
                       expect_json: bool = True, max_retries: int = 4) -> Any:
    """
    Try all bases with backoff. Raises last error on failure.
    """
    params = params or {}
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        for base in BINANCE_PUBLIC_BASES:
            url = base + path
            try:
                r = SESSION.get(url, params=params, timeout=SESSION_TIMEOUT)
                if r.status_code == 200:
                    return r.json() if expect_json else r.text
                # Many failures are 400/451 when params missing/wrong, log and continue
                log.info(f"[analyses] {url} http {r.status_code} (try {attempt})")
                last_err = RuntimeError(f"http {r.status_code} @ {url}")
            except Exception as e:
                last_err = e
                log.info(f"[analyses] GET fail: {url} err={e} (try {attempt})")
        # gentle backoff
        time.sleep(1.0 * attempt)
    assert last_err is not None
    raise last_err


def get_all_24h() -> List[Dict[str, Any]]:
    """
    Prefer pulling all tickers at once (array). If that fails, fall back to data-api mirror.
    """
    # Primary: /api/v3/ticker/24hr with no symbol returns a list for all symbols
    try:
        data = _get_with_rotation("/api/v3/ticker/24hr", params={})
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Fallback: data-api mirror path (same semantics)
    data = _get_with_rotation("/api/v3/ticker/24hr", params={})
    return data if isinstance(data, list) else []


def get_klines(symbol: str, interval: str, limit: int = 96) -> List[List[Any]]:
    """
    Klines with explicit symbol to avoid 1102 errors.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return _get_with_rotation("/api/v3/klines", params=params)


# --------------------------- Helpers: math ------------------------------------
def donchian_breakout(highs: np.ndarray, lookback: int) -> float:
    """
    Highest high of prior 'lookback' bars (exclude the current bar).
    """
    if highs.size <= lookback:
        return np.nan
    return float(np.max(highs[-(lookback + 1):-1]))


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    return tr


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, length: int) -> float:
    tr = true_range(h, l, c)
    if tr.size < length:
        return float(np.nan)
    # Wilder's smoothing (EMA-like)
    alpha = 1.0 / length
    a = tr[:length].mean()
    for x in tr[length:]:
        a = a + alpha * (x - a)
    return float(a)


def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


# --------------------------- Data classes -------------------------------------
@dataclass
class Candidate:
    ticker: str
    symbol: str
    entry: float
    stop: float
    atr: float
    breakout_ref: float
    last: float
    volume24h: float
    gain8h_pct: float
    rationale: str


# --------------------------- Scanner ------------------------------------------
def select_universe(mapping: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter mapping to symbols tradable vs USDT, avoid leveraged tokens, cap size.
    """
    uni: List[Dict[str, Any]] = []
    for row in mapping:
        rev = row.get("revolut_ticker", "").upper()
        sym = row.get("binance_symbol", "").upper()
        if not rev or not sym:
            continue
        # We want USDT pairs for liquidity/consistency
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]  # strip USDT
        # exclude known leveraged/margined suffixes
        if any(x in base for x in ("UP", "DOWN", "BEAR", "BULL")):
            continue
        uni.append({"revolut_ticker": rev, "binance_symbol": sym, "base": base})
    # de-dup & cap
    seen = set()
    uniq = []
    for r in uni:
        if r["binance_symbol"] not in seen:
            uniq.append(r)
            seen.add(r["binance_symbol"])
        if len(uniq) >= MAX_SYMBOLS:
            break
    log.info(f"[analyses] universe: {len(uniq)}")
    return uniq


def load_all_24h_snapshot() -> Dict[str, Dict[str, Any]]:
    snap = {}
    arr = get_all_24h()
    for item in arr:
        sym = item.get("symbol")
        if not sym:
            continue
        # normalize fields we use
        try:
            vol_quote = float(item.get("quoteVolume", 0.0))
            last_price = float(item.get("lastPrice", 0.0))
            snap[sym.upper()] = {
                "last": last_price,
                "quoteVolume": vol_quote,
                "priceChangePercent": float(item.get("priceChangePercent", 0.0)),
                "highPrice": float(item.get("highPrice", 0.0)),
                "lowPrice": float(item.get("lowPrice", 0.0)),
            }
        except Exception:
            continue
    return snap


def build_candidates(universe: List[Dict[str, Any]],
                     snap24h: Dict[str, Dict[str, Any]]) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """
    Scan each symbol’s recent 15m bars over ~8h to catch sustained moves.
    """
    debug_rows: List[Dict[str, Any]] = []
    cands: List[Candidate] = []

    bars_needed = int(math.ceil((LOOKBACK_HOURS * 60) / 15.0)) + BREAKOUT_LOOKBACK + ATR_LEN + 2
    bars_needed = max(bars_needed, 60)
    for i, row in enumerate(universe, 1):
        sym = row["binance_symbol"]
        base = row["base"]
        snap = snap24h.get(sym, {})
        volq = float(snap.get("quoteVolume", 0.0))
        if volq < MIN_VOLUME_USDT:
            debug_rows.append({"symbol": sym, "reason": "vol_filter", "quoteVolume": volq})
            continue

        try:
            kl = get_klines(sym, CANDLE_INTERVAL, limit=min(1000, bars_needed))
        except Exception as e:
            debug_rows.append({"symbol": sym, "reason": f"klines_fail:{e}"})
            continue

        if len(kl) < ATR_LEN + BREAKOUT_LOOKBACK + 5:
            debug_rows.append({"symbol": sym, "reason": "not_enough_bars", "bars": len(kl)})
            continue

        # columns: [open_time, open, high, low, close, volume, close_time, ...]
        o = np.array([float(x[1]) for x in kl])
        h = np.array([float(x[2]) for x in kl])
        l = np.array([float(x[3]) for x in kl])
        c = np.array([float(x[4]) for x in kl])

        last = float(c[-1])
        a = atr(h, l, c, ATR_LEN)
        if not np.isfinite(a) or a <= 0:
            debug_rows.append({"symbol": sym, "reason": "atr_nan_or_zero"})
            continue

        # 8h momentum from the close ~8h ago
        bars_8h = int(round((LOOKBACK_HOURS * 60) / 15.0))
        if bars_8h >= len(c):
            bars_8h = len(c) - 1
        ref_close = float(c[-bars_8h]) if bars_8h > 0 else float(c[0])
        gain8h = pct_change(last, ref_close)

        # Donchian breakout (exclude current bar)
        ref_high = donchian_breakout(h, BREAKOUT_LOOKBACK)
        is_breakout = np.isfinite(ref_high) and last > ref_high

        rationale = []
        if is_breakout:
            rationale.append(f"15m Donchian({BREAKOUT_LOOKBACK}) breakout")
        if gain8h >= 20:
            rationale.append(f"+{gain8h:.1f}% over ~{LOOKBACK_HOURS:.0f}h")
        if not rationale:
            debug_rows.append({"symbol": sym, "reason": "no_trigger", "gain8h_pct": gain8h})
            continue

        # breakout bar is the last bar that set ref_high. Approx: recent max
        ref_window = h[-(BREAKOUT_LOOKBACK + 1):-1]
        ref_idx = int(np.argmax(ref_window))
        breakout_bar_low = float(l[-(BREAKOUT_LOOKBACK + 1):-1][ref_idx])

        entry = max(last, ref_high)
        stop = breakout_bar_low

        cands.append(
            Candidate(
                ticker=row["revolut_ticker"],
                symbol=sym,
                entry=float(round(entry, 8)),
                stop=float(round(stop, 8)),
                atr=float(round(a, 8)),
                breakout_ref=float(round(ref_high, 8)) if np.isfinite(ref_high) else float("nan"),
                last=float(round(last, 8)),
                volume24h=volq,
                gain8h_pct=float(round(gain8h, 3)),
                rationale=", ".join(rationale),
            )
        )
        debug_rows.append({
            "symbol": sym,
            "last": last,
            "entry": entry,
            "stop": stop,
            "atr": a,
            "gain8h_pct": gain8h,
            "note": "candidate"
        })

    # Sort candidates: highest 8h gain then 24h volume
    cands.sort(key=lambda x: (x.gain8h_pct, x.volume24h), reverse=True)
    return cands, debug_rows


# --------------------------- Signals builder ----------------------------------
def signals_from_candidates(cands: List[Candidate]) -> Dict[str, Any]:
    """
    If at least one valid candidate -> signals.type = "C"
    If a very strong breakout (>= +35% and ATR tight) -> "B" for the top one.
    """
    if not cands:
        return {"type": "H", "note": "No candidates — Hold"}

    # define "B" breakout threshold
    top = cands[0]
    strong = (top.gain8h_pct >= 35.0) and (top.entry > top.breakout_ref) and (top.entry - top.stop > 0.2 * top.atr)

    if strong and (top.ticker not in UNTOUCHABLE):
        # One decisive breakout to act now
        return {
            "type": "B",
            "ticker": top.ticker,
            "symbol": top.symbol,
            "entry": top.entry,
            "stop": top.stop,
            "atr": top.atr,
            "T1": float(round(top.entry + 0.8 * top.atr, 8)),
            "T2": float(round(top.entry + 1.5 * top.atr, 8)),
            "trail_after_R": 1.0,
            "note": top.rationale,
        }

    # Otherwise list candidates
    return {
        "type": "C",
        "candidates": [
            {
                "ticker": x.ticker,
                "symbol": x.symbol,
                "entry": x.entry,
                "stop": x.stop,
                "atr": x.atr,
                "T1": float(round(x.entry + 0.8 * x.atr, 8)),
                "T2": float(round(x.entry + 1.5 * x.atr, 8)),
                "gain8h_pct": x.gain8h_pct,
                "rationale": x.rationale,
            }
            for x in cands
            if x.ticker not in UNTOUCHABLE
        ]
    }


# --------------------------- Main runner --------------------------------------
def main() -> int:
    t0 = time.time()
    ensure_dir(OUT_DIR)

    # Regime stub (extend with your own macro filters if needed)
    regime = {"ok": True, "note": "default-ok"}

    # Load mapping & universe
    mapping = load_mapping()
    universe = select_universe(mapping)

    # 24h snapshot (volume / last sanity checks)
    snap24h = load_all_24h_snapshot()

    # Build candidates via 8h breakout scan
    cands, debug_rows = build_candidates(universe, snap24h)

    # Signals
    signals = signals_from_candidates(cands)

    # Market snapshot (concise)
    mkt = {
        "generated_utc": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval": CANDLE_INTERVAL,
        "lookback_hours": LOOKBACK_HOURS,
        "atr_len": ATR_LEN,
        "breakout_lookback": BREAKOUT_LOOKBACK,
        "universe_size": len(universe),
        "scanned_symbols": len(universe),
    }

    # Summary (what the consumer expects)
    summary = {
        "status": "ok",
        "regime": regime,
        "signals": signals,
        "meta": {
            "tz": "UTC",
            "generated_at": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rules": {
                "untouchable": sorted(list(UNTOUCHABLE)),
                "do_not_rotate": sorted(list(DO_NOT_ROTATE)),
                "stop_rule": "breakout bar low",
                "trail_after_R": 1.0,
            },
        },
    }

    # Write artifacts
    json_dump(summary, os.path.join(OUT_DIR, "summary.json"))
    json_dump(signals, os.path.join(OUT_DIR, "signals.json"))
    json_dump(mkt, os.path.join(OUT_DIR, "market_snapshot.json"))
    json_dump({
        "debug": debug_rows[:1000],  # cap size
    }, os.path.join(OUT_DIR, "debug_scan.json"))

    # Run stats
    elapsed = time.time() - t0
    stats = {
        "started": int(t0),
        "finished": int(time.time()),
        "elapsed_sec": round(elapsed, 2),
        "candidates": len(cands),
        "signals_type": signals.get("type"),
    }
    json_dump(stats, os.path.join(OUT_DIR, "run_stats.json"))

    log.info(f"[analyses] wrote summary with {len(cands)} candidates in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log.error(f"[analyses] FATAL: {e}")
        # Emit minimal summary so downstream steps don't crash hard
        fail_summary = {
            "status": "error",
            "error": str(e),
            "signals": {"type": "H", "note": "Error — Hold"},
            "meta": {"generated_at": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        }
        ensure_dir(OUT_DIR)
        json_dump(fail_summary, os.path.join(OUT_DIR, "summary.json"))
        raise