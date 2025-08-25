#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses
--------
Generates swing breakout signals and fast 15m momentum candidates.
Outputs: public_runs/latest/summary.json

Rules respected:
- DOT is staked/untouchable; ETH & DOT are not rotated
- Use breakout bar low as stop
- Trail after +1R (communicated via output notes)
"""

import os
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # fallback handled below

# Lightweight deps only
import requests  # used only to detect simple prices if ccxt unavailable

# ccxt is our primary OHLCV source
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # handled gracefully


# -----------------------------
# Environment / configuration
# -----------------------------

def _get_env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v)
    except Exception:
        return default

def _get_env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(v)
    except Exception:
        return default

def _get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


# Universe & thresholds (safe defaults; can be overridden in workflow env)
MIN_VOL24_USD      = _get_env_float("MIN_VOL24_USD", 25_000_000.0)
MIN_MCAP_USD       = _get_env_float("MIN_MCAP_USD", 50_000_000.0)

FAST_MOM_ENABLED   = _get_env_bool("FAST_MOM_ENABLED", True)
FAST_MOM_TIMEFRAME = os.getenv("FAST_MOM_TIMEFRAME", "15m")
FAST_MOM_MIN_MOVE  = _get_env_float("FAST_MOM_MIN_MOVE_PCT", 8.0)   # % on last bar
FAST_MOM_MIN_RVOL  = _get_env_float("FAST_MOM_MIN_RVOL", 3.0)
FAST_MOM_MAX_SPREAD= _get_env_float("FAST_MOM_MAX_SPREAD_PCT", 0.6)
FAST_MOM_RISK_PCT  = _get_env_float("FAST_MOM_RISK_PCT", 0.6) / 100.0
FAST_MOM_CAP_PCT   = _get_env_float("FAST_MOM_CAP_PCT", 10.0) / 100.0
REGIME_GATE_FAST   = _get_env_bool("REGIME_GATE_FAST", False)

# Swing config
SWING_LOOKBACK_HIGH    = _get_env_int("SWING_LOOKBACK_HIGH", 20)     # 20D breakout
ATR_PERIOD             = _get_env_int("ATR_PERIOD", 14)
RISK_PCT_SWING         = _get_env_float("RISK_PCT_SWING", 1.2) / 100.0

# Regime cap (swing sizing cap gates)
CAP_OK                 = _get_env_float("CAP_OK", 0.60)
CAP_WEAK               = _get_env_float("CAP_WEAK", 0.30)

# Repo paths
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "public_runs" / "latest"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Untouchables / not rotated
UNTOUCHABLE = {"DOT"}
NOT_ROTATED = {"ETH", "DOT"}


# -----------------------------
# Helpers
# -----------------------------

def now_prague_iso() -> str:
    try:
        tz = ZoneInfo("Europe/Prague") if ZoneInfo else None
    except Exception:
        tz = None
    if tz:
        return datetime.now(tz).isoformat()
    return datetime.now(timezone.utc).astimezone().isoformat()

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    k = 2.0 / (period + 1.0)
    out = []
    s = series[0]
    out.append(s)
    for v in series[1:]:
        s = v * k + s * (1 - k)
        out.append(s)
    return out

def true_range(h: List[float], l: List[float], c: List[float]) -> List[float]:
    tr = []
    for i in range(len(c)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
    return tr

def atr(h: List[float], l: List[float], c: List[float], period: int) -> List[float]:
    tr = true_range(h, l, c)
    return ema(tr, period)

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0

def safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


# -----------------------------
# Market data via ccxt
# -----------------------------

class Market:
    def __init__(self):
        self.exchanges = []
        if ccxt:
            # Try Binance → OKX → Kucoin (all public endpoints)
            for ex in ("binance", "okx", "kucoin"):
                try:
                    inst = getattr(ccxt, ex)({"enableRateLimit": True})
                    self.exchanges.append(inst)
                except Exception:
                    pass

    def best_symbol(self, ticker: str) -> Optional[str]:
        # We prefer /USDT; fallback /USD
        candidates = [f"{ticker}/USDT", f"{ticker}/USD"]
        for ex in self.exchanges:
            for s in candidates:
                try:
                    m = ex.market(s)
                    if m and not m.get("active") is False:
                        return s
                except Exception:
                    continue
        return None

    def fetch_ohlcv(self, ticker: str, timeframe: str, limit: int = 210) -> Optional[List[List[float]]]:
        symbol = self.best_symbol(ticker)
        if not symbol:
            return None
        for ex in self.exchanges:
            try:
                return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            except Exception:
                continue
        return None

    def spread_pct(self, ticker: str) -> Optional[float]:
        symbol = self.best_symbol(ticker)
        if not symbol:
            return None
        for ex in self.exchanges:
            try:
                ob = ex.fetch_order_book(symbol, limit=5)
                bid = ob["bids"][0][0] if ob["bids"] else None
                ask = ob["asks"][0][0] if ob["asks"] else None
                if bid and ask:
                    mid = 0.5 * (bid + ask)
                    return (ask - bid) / mid * 100.0
            except Exception:
                continue
        return None


# -----------------------------
# Universe (simple & robust)
# -----------------------------

def load_universe() -> List[str]:
    """
    Build a conservative, Revolut-friendly universe. If a mapping file exists
    in the repo (mapping/*.json), we use its tickers. Otherwise, use a sensible
    default list that includes majors + midcaps (ACS included).
    """
    # Try to read mapping files if present
    tickers: List[str] = []
    mapping_dirs = [ROOT / "mapping", ROOT / "data", ROOT / "configs"]
    for d in mapping_dirs:
        if d.exists():
            for p in d.glob("*.json"):
                try:
                    obj = json.loads(p.read_text())
                    # attempt to extract tickers lists if known format
                    if isinstance(obj, dict):
                        if "tickers" in obj and isinstance(obj["tickers"], list):
                            tickers.extend([str(x).upper() for x in obj["tickers"]])
                        elif "mapping" in obj and isinstance(obj["mapping"], dict):
                            tickers.extend([str(k).upper() for k in obj["mapping"].keys()])
                except Exception:
                    pass

    if tickers:
        tickers = sorted(set(tickers))
    else:
        # Fallback default (includes ACS as requested)
        tickers = [
            "BTC","ETH","SOL","BNB","XRP","ADA","AVAX","DOGE","TRX","LINK",
            "MATIC","IMX","OP","ARB","DOT","NEAR","APT","ATOM","SUI","SEI",
            "TIA","JUP","PYTH","INJ","RUNE","FTM","SAND","AXS","APE","AAVE",
            "GRT","RNDR","WIF","BONK","SHIB","TIA","NEON","SHDW","ACS"
        ]
        tickers = sorted(set(tickers))

    # Never include untouchables for trading outputs (but keep for regime metrics if needed)
    return tickers


# -----------------------------
# Regime (BTC daily vs EMA50)
# -----------------------------

def compute_regime(mkt: Market) -> Dict[str, Any]:
    data = mkt.fetch_ohlcv("BTC", timeframe="1d", limit=120)
    ok = False
    reason = "insufficient-data"
    if data and len(data) > 60:
        closes = [x[4] for x in data]
        ema50 = ema(closes, 50)
        if closes[-1] > ema50[-1]:
            ok = True
            reason = "price>ema50"
        else:
            ok = False
            reason = "price<=ema50"
    return {"ok": ok, "reason": reason}


# -----------------------------
# Swing breakout logic (daily)
# -----------------------------

def swing_breakouts(mkt: Market, tickers: List[str]) -> List[Dict[str, Any]]:
    picks: List[Dict[str, Any]] = []
    for t in tickers:
        if t in UNTOUCHABLE:
            continue  # never trade DOT
        # Non-rotated filtering only matters when rotating between positions; okay to still signal
        data = mkt.fetch_ohlcv(t, timeframe="1d", limit=max(SWING_LOOKBACK_HIGH + 5, ATR_PERIOD + 5))
        if not data or len(data) < SWING_LOOKBACK_HIGH + 2:
            continue

        o = [x[1] for x in data]
        h = [x[2] for x in data]
        l = [x[3] for x in data]
        c = [x[4] for x in data]

        prev_high = max(h[-(SWING_LOOKBACK_HIGH+1):-1])  # prior N-day high, exclude last bar
        last_close = c[-1]
        last_low = l[-1]
        atr_list = atr(h, l, c, ATR_PERIOD)
        last_atr = atr_list[-1]

        # Breakout if last close > previous N-day high
        if last_close > prev_high:
            entry = last_close
            stop = last_low  # breakout bar low as stop (rule)
            if stop >= entry:  # guard
                continue
            t1 = entry + 0.8 * last_atr
            t2 = entry + 1.5 * last_atr
            picks.append({
                "ticker": t,
                "entry": round(entry, 8),
                "stop": round(stop, 8),
                "atr": round(last_atr, 8),
                "t1": round(t1, 8),
                "t2": round(t2, 8),
                "timeframe": "1d",
                "type": "breakout"
            })

    # Rank by distance above prior high (proxy = (entry/stop) or simply ATR multiple)
    picks.sort(key=lambda x: (x["entry"] - x["stop"]) / max(1e-9, x["atr"]), reverse=True)
    return picks


# -----------------------------
# Fast 15m momentum scanner
# -----------------------------

def fast_candidates(mkt: Market, tickers: List[str]) -> List[Dict[str, Any]]:
    if not FAST_MOM_ENABLED:
        return []

    cands: List[Dict[str, Any]] = []
    for t in tickers:
        if t in UNTOUCHABLE:
            continue

        # Spread guard (skip if too wide)
        spr = mkt.spread_pct(t)
        if spr is not None and spr > FAST_MOM_MAX_SPREAD:
            continue

        data = mkt.fetch_ohlcv(t, timeframe=FAST_MOM_TIMEFRAME, limit=max(40, ATR_PERIOD + 5))
        if not data or len(data) < 30:
            continue

        o = [x[1] for x in data]
        h = [x[2] for x in data]
        l = [x[3] for x in data]
        c = [x[4] for x in data]
        v = [x[5] for x in data]

        last_close = c[-1]
        prev_close = c[-2]
        chg15 = pct(last_close, prev_close)

        avg_vol20 = sum(v[-20:]) / 20.0 if len(v) >= 20 else (sum(v) / len(v))
        rvol = (v[-1] / max(1.0, avg_vol20)) if avg_vol20 else 0.0

        ema9 = ema(c, 9)[-1]
        ema20 = ema(c, 20)[-1]

        if chg15 >= FAST_MOM_MIN_MOVE and rvol >= FAST_MOM_MIN_RVOL and last_close >= ema9 >= ema20:
            atr_list = atr(h, l, c, ATR_PERIOD)
            last_atr = atr_list[-1]
            entry = last_close
            stop = l[-1]  # breakout bar low on 15m
            if stop >= entry:
                continue
            t1 = entry + 0.8 * last_atr
            t2 = entry + 1.5 * last_atr
            cands.append({
                "ticker": t,
                "entry": round(entry, 8),
                "stop": round(stop, 8),
                "atr": round(last_atr, 8),
                "t1": round(t1, 8),
                "t2": round(t2, 8),
                "timeframe": FAST_MOM_TIMEFRAME,
                "reason": "15m-rvol-spike",
                "rvol": round(rvol, 3),
                "chg_pct": round(chg15, 3),
                "spread_pct": None if spr is None else round(spr, 3),
                "cap_pct": FAST_MOM_CAP_PCT,
                "risk_pct": FAST_MOM_RISK_PCT
            })

    # Rank by rvol, then % change
    cands.sort(key=lambda x: (x["rvol"], x["chg_pct"]), reverse=True)
    return cands


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    start = time.time()
    mkt = Market()
    tickers = load_universe()

    regime = compute_regime(mkt)

    swing = swing_breakouts(mkt, tickers)
    fast = fast_candidates(mkt, tickers)

    signals: Dict[str, Any] = {}
    if swing:
        signals["type"] = "B"   # breakout
        signals["picks"] = swing[:3]  # top 3
    elif fast:
        # keep swing type separate; fast candidates go in their own section
        signals["type"] = "C"   # candidates only (no swing breakouts)
        signals["picks"] = []
    else:
        signals["type"] = "H"   # hold/wait
        signals["picks"] = []

    # Output summary
    summary: Dict[str, Any] = {
        "status": "ok",
        "timestamp_prague": now_prague_iso(),
        "regime": regime,
        "signals": signals,
        "candidates": swing,           # keep for backward compatibility
        "fast_candidates": fast,       # NEW — this is what catches ACS-like moves
        "notes": [
            "DOT is staked and untouchable; ETH and DOT are not rotated.",
            "Use breakout bar low as stop; trail after +1R.",
            "Swing sizing cap: 60% when regime.ok else 30%.",
            "Fast-momentum trades use smaller risk and cap (configurable)."
        ],
        "config": {
            "SWING_LOOKBACK_HIGH": SWING_LOOKBACK_HIGH,
            "ATR_PERIOD": ATR_PERIOD,
            "RISK_PCT_SWING": RISK_PCT_SWING,
            "FAST_MOM_ENABLED": FAST_MOM_ENABLED,
            "FAST_MOM_TIMEFRAME": FAST_MOM_TIMEFRAME,
            "FAST_MOM_MIN_MOVE_PCT": FAST_MOM_MIN_MOVE,
            "FAST_MOM_MIN_RVOL": FAST_MOM_MIN_RVOL,
            "FAST_MOM_MAX_SPREAD_PCT": FAST_MOM_MAX_SPREAD,
            "FAST_MOM_RISK_PCT": FAST_MOM_RISK_PCT,
            "FAST_MOM_CAP_PCT": FAST_MOM_CAP_PCT,
            "REGIME_GATE_FAST": REGIME_GATE_FAST,
            "CAP_OK": CAP_OK,
            "CAP_WEAK": CAP_WEAK
        },
        "runtime_sec": round(time.time() - start, 3)
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    # Optional: thin companion files
    (OUTDIR / "signals.json").write_text(json.dumps({"signals": signals}, indent=2))
    (OUTDIR / "run_stats.json").write_text(json.dumps({"runtime_sec": summary["runtime_sec"]}, indent=2))

    print(f"[analyses] wrote {OUTDIR/'summary.json'} (swing picks: {len(swing)}, fast: {len(fast)})")


if __name__ == "__main__":
    main()