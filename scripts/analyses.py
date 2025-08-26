#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import pathlib
import statistics
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from time import perf_counter

import requests
import pandas as pd

# ---- local helpers from your repo ----
# Must exist: scripts/marketdata.py with get_btc_5m_klines, ema, vwap
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------- Paths / constants ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
PUB_LATEST = ROOT / "public_runs" / "latest"
for p in (ARTIFACTS, DATA, PUB_LATEST):
    p.mkdir(parents=True, exist_ok=True)

SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNAL_PATH = DATA / "signals.json"
RUN_STATS = ARTIFACTS / "run_stats.json"
DEBUG_PATH = ARTIFACTS / "debug_scan.json"
SUMMARY_PATH = PUB_LATEST / "summary.json"
DEBUG_LOG: List[Dict[str, Any]] = []

# ---------- Trading config ----------
VOL_USD_MIN = 8_000_000        # 24h volume threshold
MAX_SPREAD = 0.005              # ≤ 0.5%
RISK_PCT = 0.012                # 1.2% risk
STRONG_ALLOC = 0.60
WEAK_ALLOC = 0.30
EQUITY_FLOOR = 40000.0          # updated floor

EXCLUDE_ROTATION = {"ETH", "DOT"}  # DOT staked; ETH and DOT not rotated

# ---------- Binance only (with symbol param) ----------
BINANCE_BASE = "https://data-api.binance.vision"
HEADERS = {
    "User-Agent": "rev-analyses/2.0 (+https://github.com/armendq/revolut_crypto_mapping)",
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

# ---------- Spike detector params ----------
H_INTERVAL = "1h"
H_LIMIT = 120
WINDOW_H = 8
MIN_PCT_RISE = 0.20    # +20% over 8h
VOL_MULT = 1.8
ATR_LEN = 14

# ---------- Breakout detector params ----------
M1_INTERVAL = "1m"
M1_LIMIT = 120

blocked_451 = False  # flipped if HTTP 451 (should not happen on data-api, but keep)

# ---------- small utilities ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(event: str, **fields):
    DEBUG_LOG.append({"t": _now_iso(), "event": event, **fields})

def write_json(path: pathlib.Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _as_float(x: Any) -> float:
    if hasattr(x, "iloc"):
        return float(x.iloc[-1])
    return float(x)

# ---------- HTTP ----------
def get_json(path: str, params: Dict[str, Any]) -> Any:
    global blocked_451
    url = f"{BINANCE_BASE}{path}"
    last_err = None
    for i in range(1, RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                log("http_ok", url=url, params=params, try_num=i)
                return r.json()
            if r.status_code == 451:
                blocked_451 = True
            last_err = f"HTTP {r.status_code}"
            log("http_non_200", url=url, params=params, status=r.status_code, try_num=i)
        except Exception as e:
            last_err = str(e)
            log("http_exc", url=url, params=params, err=last_err, try_num=i)
        time.sleep(BACKOFF ** (i - 1))
    raise RuntimeError(f"GET failed: {url} err={last_err}")

# ---------- Binance data helpers ----------
def bn_ticker_24h(symbol: str) -> Optional[Dict[str, float]]:
    try:
        d = get_json("/api/v3/ticker/24hr", {"symbol": symbol})
        last = float(d.get("lastPrice", 0.0))
        vol_base = float(d.get("volume", 0.0))
        bid = float(d.get("bidPrice", 0.0))
        ask = float(d.get("askPrice", 0.0))
        vol_usd = vol_base * last
        spread = 0.0
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        if mid > 0:
            spread = (ask - bid) / mid
        return {"price": last, "vol_usd": vol_usd, "bid": bid, "ask": ask, "spread": spread}
    except Exception as e:
        log("bn_ticker_24h_fail", symbol=symbol, err=str(e))
        return None

def bn_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    # Returns list of klines arrays:
    # [ openTime, open, high, low, close, volume, ... ]
    arr = get_json("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(arr, list) or not arr:
        raise RuntimeError("empty_klines")
    return arr

# ---------- TA helpers ----------
def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines_1m(bars: List[Dict[str, float]], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]
        l = bars[-i]["l"]
        pc = bars[-i - 1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

def calc_atr_from_raw(kl: List[List[Any]], length: int = ATR_LEN) -> float:
    trs: List[float] = []
    for i in range(1, len(kl)):
        pc = float(kl[i - 1][4])
        hi = float(kl[i][2])
        lo = float(kl[i][3])
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    if len(trs) < length:
        return float("nan")
    return statistics.fmean(trs[-length:])

def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

# ---------- Breakout rules (1m) ----------
def aggressive_breakout_1m(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]
    prev = bars_1m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    if pct < 0.018 or pct > 0.04:
        return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]])
    rvol = (last["v"] / vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15}

def micro_pullback_ok_1m(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] == 0 or last["c"] == 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "stop": last["l"]}
    return None

# ---------- Spike rules (1h) ----------
def detect_spike_8h(kl: List[List[Any]]) -> Dict[str, Any]:
    need = ATR_LEN + WINDOW_H + 5
    if len(kl) < need:
        return {}
    closes = [float(r[4]) for r in kl]
    lows = [float(r[3]) for r in kl]
    vols = [float(r[5]) for r in kl]
    last_close = closes[-1]
    ref_close = closes[-(WINDOW_H + 1)]
    if ref_close <= 0:
        return {}
    pct = (last_close - ref_close) / ref_close
    last8_vol = sum(vols[-WINDOW_H:])
    prev8_vol = sum(vols[-(2 * WINDOW_H):-WINDOW_H]) + 1e-12
    vol_mult = last8_vol / prev8_vol
    if pct < MIN_PCT_RISE or vol_mult < VOL_MULT:
        return {}
    atr = calc_atr_from_raw(kl, ATR_LEN)
    if not math.isfinite(atr) or atr <= 0:
        return {}
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

# ---------- Regime ----------
def check_regime() -> Dict[str, Any]:
    bars = get_btc_5m_klines()
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        return {"ok": False, "reason": "no-btc-5m"}
    close = bars["close"]
    try:
        last = _as_float(close.iloc[-1])
        vw = _as_float(vwap(bars))
        e9 = _as_float(ema(close, span=9))
    except Exception:
        return {"ok": False, "reason": "calc-error"}
    ok = (last > vw) and (last > e9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-or-ema",
            "last": last, "vwap": vw, "ema9": e9}

# ---------- Universe / mapping ----------
def load_revolut_mapping() -> List[Dict[str, Any]]:
    path = DATA / "revolut_mapping.json"
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for m in obj:
            if isinstance(m, dict):
                out.append(m)
            elif isinstance(m, str):
                out.append({"ticker": m})
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                vv = dict(v)
                vv.setdefault("ticker", k)
                out.append(vv)
            else:
                out.append({"ticker": k})
    return out

def best_binance_symbol(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("binance_symbol"):
        return str(entry["binance_symbol"]).upper()
    if entry.get("binance"):
        return str(entry["binance"]).upper()
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# ---------- Sizing ----------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = STRONG_ALLOC if strong_regime else WEAK_ALLOC
    usd_cap = equity * cap
    return max(0.0, min(usd, usd_cap, cash))

# ---------- MAIN ----------
def main():
    t_run0 = perf_counter()
    log("run_start", ts=_now_iso())

    equity = float(os.getenv("EQUITY", "41000") or "41000")
    cash = float(os.getenv("CASH", "32000") or "32000")

    snapshot = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "floor": EQUITY_FLOOR,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": []
    }

    # Capital preservation guard
    buffer_ok = (equity - EQUITY_FLOOR) >= 1000.0
    if not buffer_ok or equity <= (EQUITY_FLOOR + 500.0):  # hard guard near floor
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity_near_floor"
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."})
        write_json(RUN_STATS, {"elapsed_ms": int((perf_counter() - t_run0) * 1000), "time": _now_iso()})
        write_json(DEBUG_PATH, DEBUG_LOG)
        # also publish summary
        write_json(SUMMARY_PATH, {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "equity": equity,
            "cash": cash,
            "regime": {"ok": False, "reason": "capital-guard"},
            "candidates": [],
            "signals": {"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."},
            "run_stats": {"elapsed_ms": int((perf_counter() - t_run0) * 1000), "time": _now_iso()}
        })
        print("Raise cash now.")
        return

    # 1) Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong = bool(regime.get("ok", False))

    # 2) Universe build
    mapping = load_revolut_mapping()
    universe = []
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if not tkr:
            continue
        sym = best_binance_symbol(m)
        if not sym:
            continue
        if not sym.endswith("USDT"):
            continue
        # Exclude ETH, DOT from rotation per spec
        if tkr in EXCLUDE_ROTATION:
            continue
        # Meta from 24h ticker to filter
        meta = bn_ticker_24h(sym)
        if not meta:
            continue
        spr = meta.get("spread", None)
        vol_usd = float(meta.get("vol_usd") or 0.0)
        if spr is None:
            continue
        if vol_usd >= VOL_USD_MIN and spr <= MAX_SPREAD:
            u = {
                "ticker": tkr,
                "symbol": sym,
                "price": float(meta["price"]),
                "bid": float(meta["bid"]),
                "ask": float(meta["ask"]),
                "spread": spr,
                "vol_usd": vol_usd,
                "src": "binance"
            }
            universe.append(u)

    snapshot["universe_count"] = len(universe)

    # 3) Candidate generation
    candidates: List[Dict[str, Any]] = []
    for u in universe:
        sym = u["symbol"]

        # 1m bars for breakout
        try:
            kl1 = bn_klines(sym, M1_INTERVAL, M1_LIMIT)
            bars_1m = [
                {"ts": r[0], "o": float(r[1]), "h": float(r[2]),
                 "l": float(r[3]), "c": float(r[4]), "v": float(r[5])}
                for r in kl1
            ]
        except Exception as e:
            log("klines_1m_fail", symbol=sym, err=str(e))
            bars_1m = []

        # 1h bars for spike
        try:
            klh = bn_klines(sym, H_INTERVAL, H_LIMIT)
        except Exception as e:
            log("klines_1h_fail", symbol=sym, err=str(e))
            klh = []

        br = aggressive_breakout_1m(bars_1m) if bars_1m else None
        pb = micro_pullback_ok_1m(bars_1m) if bars_1m else None
        atr1m = atr_from_klines_1m(bars_1m, period=14) if bars_1m else 0.0

        sp = detect_spike_8h(klh) if klh else {}

        # Build candidate only if at least one signal exists
        if (br and pb and atr1m > 0) or sp:
            # Combined scoring
            br_score = (br["rvol"] * (1.0 + br["pct"])) if br else 0.0
            sp_score = (sp.get("pct8h", 0.0) * 0.01) * max(1.0, sp.get("vol_mult", 0.0)) if sp else 0.0
            score = br_score + sp_score

            # Use spike levels if present; else 1m pullback levels
            if sp:
                entry = float(sp["entry"])
                stop = float(sp["stop"])
                atr_used = float(sp["atr"])
                t1 = float(sp["t1"])
                t2 = float(sp["t2"])
                source = "spike_8h"
            else:
                entry = float(pb["entry"])
                stop = float(pb["stop"])
                atr_used = float(atr1m)
                t1 = entry + 0.8 * atr_used
                t2 = entry + 1.5 * atr_used
                source = "breakout_1m"

            candidates.append({
                "ticker": u["ticker"],
                "symbol": sym,
                "src": source,
                "entry": round(entry, 8),
                "stop": round(stop, 8),
                "atr": round(atr_used, 8),
                "t1": round(t1, 8),
                "t2": round(t2, 8),
                "rvol": round(br["rvol"], 2) if br else None,
                "m1_pct": round(br["pct"] * 100.0, 2) if br else None,
                "pct8h": sp.get("pct8h"),
                "vol_mult": sp.get("vol_mult"),
                "score": round(score, 6),
            })

        time.sleep(0.02)  # mild pacing

    candidates.sort(key=lambda x: x["score"], reverse=True)
    snapshot["candidates"] = candidates[:5]  # keep top 5 in snapshot

    # 4) Signals and outputs
    if blocked_451:
        snapshot["regime"] = {"ok": False, "reason": "binance-451-block"}

    if not strong or not candidates:
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "C", "text": "Hold and wait."})
        write_json(DEBUG_PATH, DEBUG_LOG)
        run_ms = int((perf_counter() - t_run0) * 1000)
        write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": _now_iso()})
        # publish summary
        write_json(SUMMARY_PATH, {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regime": snapshot["regime"],
            "equity": equity,
            "cash": cash,
            "candidates": candidates,
            "signals": {"type": "C", "text": "Hold and wait."},
            "run_stats": {"elapsed_ms": run_ms, "time": _now_iso()}
        })
        print("Hold and wait. (No qualified candidates or weak regime.)")
        return

    # Select top
    top = candidates[0]
    entry = float(top["entry"])
    stop = float(top["stop"])
    atr_used = float(top["atr"])
    t1 = float(top["t1"])
    t2 = float(top["t2"])

    buy_usd = position_size(entry, stop, equity, cash, strong)

    plan_lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.8f}",
        f"• Target: T1 {t1:.8f}, T2 {t2:.8f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below {stop:.8f} or stall > 5 min",
        "• What to sell from portfolio (excluding ETH, DOT): Sell weakest on support break if cash needed.",
        f"• Exact USD buy amount so equity risk=1.2% and cap respected: ${buy_usd:,.2f}",
        f"• Source: {top['src']}",
    ]

    write_json(SIGNAL_PATH, {"type": "B", "text": "\n".join(plan_lines)})
    write_json(SNAP_PATH, snapshot)
    write_json(DEBUG_PATH, DEBUG_LOG)
    run_ms = int((perf_counter() - t_run0) * 1000)
    write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": _now_iso()})

    # publish summary with detailed candidates
    write_json(SUMMARY_PATH, {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": snapshot["regime"],
        "equity": equity,
        "cash": cash,
        "candidates": candidates,
        "signals": {
            "type": "B",
            "text": "\n".join(plan_lines),
            "selected": {
                "ticker": top["ticker"],
                "entry": entry,
                "stop": stop,
                "t1": t1,
                "t2": t2,
                "atr": atr_used,
                "src": top["src"],
                "buy_usd": buy_usd
            }
        },
        "run_stats": {"elapsed_ms": run_ms, "time": _now_iso()}
    })

    print("\n".join(plan_lines))


if __name__ == "__main__":
    main()