#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import pathlib
import statistics
from typing import Any, Dict, List, Optional
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
VOL_USD_MIN_24H = 8_000_000      # 24h volume threshold for universe
MAX_SPREAD = 0.005                # ≤ 0.5%
RISK_PCT = 0.012                  # 1.2% risk
STRONG_ALLOC = 0.60
WEAK_ALLOC = 0.30
EQUITY_FLOOR = 40000.0            # updated floor

EXCLUDE_ROTATION = {"ETH", "DOT"} # DOT staked; ETH and DOT not rotated

# ---------- Momentum gates ----------
# 8h momentum window using 1h bars
WIN_H = 8
MIN_PCT_8H = 12.0                 # %
MIN_RVOL_8H = 3.0                 # last 8h volume vs previous 8h
MIN_VOL_8H_USD = 1_500_000.0

# 24h momentum window using 1h bars
WIN_24H = 24
MIN_PCT_24H = 20.0                # %

ATR_LEN = 14                      # ATR period
PAUSE_SEC_PER_SYMBOL = 0.02       # gentle pacing

# ---------- Binance (strict symbol param) ----------
BINANCE_BASE = "https://data-api.binance.vision"
B_HEADERS = {
    "User-Agent": "rev-analyses/3.0 (+https://github.com/armendq/revolut_crypto_mapping)",
    "Accept": "application/json,text/plain,*/*",
}
TIMEOUT = 20
RETRIES = 4
BACKOFF = 1.6
blocked_451 = False

# ---------- Coinbase ----------
COINBASE_BASE = "https://api.exchange.coinbase.com"
C_HEADERS = {
    "User-Agent": "rev-analyses/3.0 (+https://github.com/armendq/revolut_crypto_mapping)",
    "Accept": "application/json",
}

# ---------- utils ----------
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
def http_get_json(url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> Any:
    global blocked_451
    last_err = None
    for i in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                log("http_ok", url=url, params=params or {}, try_num=i)
                return r.json()
            if r.status_code == 451:
                blocked_451 = True
            last_err = f"HTTP {r.status_code}"
            log("http_non200", url=url, status=r.status_code, try_num=i)
        except Exception as e:
            last_err = str(e)
            log("http_exc", url=url, err=last_err, try_num=i)
        time.sleep(BACKOFF ** (i - 1))
    raise RuntimeError(f"GET failed: {url} err={last_err}")

# ---------- Binance data ----------
def bn_ticker_24h(symbol: str) -> Optional[Dict[str, float]]:
    try:
        d = http_get_json(f"{BINANCE_BASE}/api/v3/ticker/24hr", B_HEADERS, {"symbol": symbol})
        last = float(d.get("lastPrice", 0.0))
        vol_base = float(d.get("volume", 0.0))
        bid = float(d.get("bidPrice", 0.0))
        ask = float(d.get("askPrice", 0.0))
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread = (ask - bid) / mid if mid > 0 else 9.99
        return {"price": last, "vol_usd": vol_base * last, "bid": bid, "ask": ask, "spread": spread}
    except Exception as e:
        log("bn_ticker_24h_fail", symbol=symbol, err=str(e))
        return None

def bn_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    return http_get_json(f"{BINANCE_BASE}/api/v3/klines", B_HEADERS, {"symbol": symbol, "interval": interval, "limit": limit})

# ---------- Coinbase data ----------
def cb_product(ticker: str) -> str:
    return f"{ticker.upper()}-USD"

def cb_book_best(ticker: str) -> Optional[Dict[str, float]]:
    try:
        pid = cb_product(ticker)
        d = http_get_json(f"{COINBASE_BASE}/products/{pid}/book", C_HEADERS, {"level": 1})
        bid = float(d["bids"][0][0])
        ask = float(d["asks"][0][0])
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread = (ask - bid) / mid if mid > 0 else 9.99
        return {"bid": bid, "ask": ask, "spread": spread}
    except Exception as e:
        log("cb_book_fail", ticker=ticker, err=str(e))
        return None

def cb_stats_24h(ticker: str) -> Optional[Dict[str, float]]:
    try:
        pid = cb_product(ticker)
        d = http_get_json(f"{COINBASE_BASE}/products/{pid}/stats", C_HEADERS)
        last = float(d.get("last", 0.0))
        vol_base = float(d.get("volume", 0.0))
        return {"price": last, "vol_usd": vol_base * last}
    except Exception as e:
        log("cb_stats_fail", ticker=ticker, err=str(e))
        return None

def cb_candles(ticker: str, granularity: int, limit: int) -> List[List[Any]]:
    # Coinbase returns [ time, low, high, open, close, volume ] (time in unix seconds)
    pid = cb_product(ticker)
    arr = http_get_json(f"{COINBASE_BASE}/products/{pid}/candles", C_HEADERS, {"granularity": granularity})
    if not isinstance(arr, list):
        return []
    arr.sort(key=lambda r: r[0])
    return arr[-limit:]

# ---------- TA ----------
def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_bars_ohlcv(bars: List[Dict[str, float]], length: int = ATR_LEN) -> float:
    if len(bars) < length + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, length + 1):
        pc = bars[-i - 1]["c"]
        hi = bars[-i]["h"]
        lo = bars[-i]["l"]
        trs.append(true_range(hi, lo, pc))
    return statistics.fmean(trs) if trs else 0.0

def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

# ---------- Signals ----------
def compute_momentum_1h(kl: List[List[Any]]) -> Dict[str, Any]:
    out = {
        "pct_8h": None, "rvol_8h": None, "usdvol_8h": None,
        "pct_24h": None
    }
    if not kl or len(kl) < max(WIN_24H + 2, ATR_LEN + 5):
        return out
    closes = [float(r[4]) for r in kl]
    lows   = [float(r[3]) for r in kl]
    vols   = [float(r[5]) for r in kl]  # base vol
    last = closes[-1]

    # 8h
    ref8 = closes[-(WIN_H + 1)]
    pct8 = (last - ref8) / ref8 if ref8 > 0 else 0.0
    last8_vol_base = sum(vols[-WIN_H:])
    prev8_vol_base = sum(vols[-(2 * WIN_H):-WIN_H]) or 1e-9
    rvol8 = last8_vol_base / prev8_vol_base

    # rough USD vol over 8h (avg close * base vol)
    avg8 = statistics.fmean(closes[-WIN_H:])
    usdvol8 = avg8 * last8_vol_base

    # 24h
    ref24 = closes[-(WIN_24H + 1)]
    pct24 = (last - ref24) / ref24 if ref24 > 0 else 0.0

    out["pct_8h"] = pct8 * 100.0
    out["rvol_8h"] = rvol8
    out["usdvol_8h"] = usdvol8
    out["pct_24h"] = pct24 * 100.0
    out["last_low_8h"] = min(lows[-WIN_H:])
    return out

def aggressive_breakout_15m(bars_15m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_15m) < 20:
        return None
    last = bars_15m[-1]
    prev = bars_15m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    # looser than 1m version; still requires thrust and follow-through
    if pct < 0.008 or pct > 0.05:
        return None
    vol_med = median([b["v"] for b in bars_15m[-16:-1]])
    rvol = (last["v"] / vol_med) if vol_med > 0 else 0.0
    if rvol < 2.5:
        return None
    hh15 = max(b["h"] for b in bars_15m[-15:])
    if last["c"] <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "entry": last["c"], "stop": last["l"]}

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

# ---------- sizing ----------
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
    t0 = perf_counter()
    log("run_start")

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
    if not buffer_ok or equity <= (EQUITY_FLOOR + 500.0):
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity_near_floor"
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."})
        run_ms = int((perf_counter() - t0) * 1000)
        write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": _now_iso()})
        write_json(DEBUG_PATH, DEBUG_LOG)
        write_json(SUMMARY_PATH, {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "equity": equity,
            "cash": cash,
            "regime": {"ok": False, "reason": "capital-guard"},
            "candidates": [],
            "signals": {"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."},
            "run_stats": {"elapsed_ms": run_ms, "time": _now_iso()}
        })
        print("Raise cash now.")
        return

    # Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong = bool(regime.get("ok", False))

    # Universe
    mapping = load_revolut_mapping()
    universe = []
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if not tkr:
            continue
        if tkr in EXCLUDE_ROTATION:
            continue

        sym = best_binance_symbol(m)
        meta = bn_ticker_24h(sym) if sym else None

        if meta and meta["vol_usd"] >= VOL_USD_MIN_24H and meta["spread"] <= MAX_SPREAD:
            universe.append({"ticker": tkr, "src": "binance", "symbol": sym, **meta})
        else:
            # try Coinbase fallback
            stats = cb_stats_24h(tkr)
            book = cb_book_best(tkr)
            if stats and book and stats["vol_usd"] >= VOL_USD_MIN_24H and book["spread"] <= MAX_SPREAD:
                universe.append({"ticker": tkr, "src": "coinbase", "symbol": cb_product(tkr),
                                 "price": stats["price"], "vol_usd": stats["vol_usd"],
                                 "bid": book["bid"], "ask": book["ask"], "spread": book["spread"]})

        time.sleep(PAUSE_SEC_PER_SYMBOL)

    snapshot["universe_count"] = len(universe)

    # Candidates
    candidates: List[Dict[str, Any]] = []
    for u in universe:
        tkr = u["ticker"]
        src = u["src"]

        # 1h candles (for 8h/24h momentum)
        if src == "binance":
            try:
                kl1h = bn_klines(u["symbol"], "1h", 120)
            except Exception as e:
                log("kl_1h_fail_bn", symbol=u["symbol"], err=str(e))
                kl1h = []
        else:
            try:
                kl1h = cb_candles(tkr, 3600, 120)
            except Exception as e:
                log("kl_1h_fail_cb", ticker=tkr, err=str(e))
                kl1h = []

        # 15m candles (for breakout/ATR/stop/targets)
        if src == "binance":
            try:
                kl15 = bn_klines(u["symbol"], "15m", 120)
            except Exception as e:
                log("kl_15m_fail_bn", symbol=u["symbol"], err=str(e))
                kl15 = []
        else:
            try:
                kl15 = cb_candles(tkr, 900, 120)
            except Exception as e:
                log("kl_15m_fail_cb", ticker=tkr, err=str(e))
                kl15 = []

        # Normalize 15m bars
        bars_15m: List[Dict[str, float]] = []
        for r in kl15:
            # Binance: [ openTime, open, high, low, close, volume, ... ]
            # Coinbase: [ time, low, high, open, close, volume ]
            if len(r) < 6:
                continue
            if src == "binance":
                bars_15m.append({"ts": r[0], "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4]), "v": float(r[5])})
            else:
                bars_15m.append({"ts": r[0] * 1000, "o": float(r[3]), "h": float(r[2]), "l": float(r[1]), "c": float(r[4]), "v": float(r[5])})

        # Momentum 1h
        mom = compute_momentum_1h(kl1h)

        mom_pass = False
        reasons = []
        if mom["pct_8h"] is not None:
            if (mom["pct_8h"] >= MIN_PCT_8H and
                (mom["rvol_8h"] or 0.0) >= MIN_RVOL_8H and
                (mom["usdvol_8h"] or 0.0) >= MIN_VOL_8H_USD):
                mom_pass = True
                reasons.append("8h_momentum")
        if not mom_pass and mom["pct_24h"] is not None:
            if mom["pct_24h"] >= MIN_PCT_24H:
                mom_pass = True
                reasons.append("24h_momentum")

        br = aggressive_breakout_15m(bars_15m) if bars_15m else None
        atr = atr_from_bars_ohlcv(bars_15m, ATR_LEN) if bars_15m else 0.0

        # Entry/stop selection:
        entry = None
        stop = None
        src_sig = None

        if mom_pass:
            # momentum-driven: entry = last 15m close; stop = last 8h lowest low or 15m bar low, tighter of the two
            if bars_15m:
                last_c = bars_15m[-1]["c"]
                stop_15m = bars_15m[-1]["l"]
                stop_8h = mom.get("last_low_8h", stop_15m)
                entry = last_c
                stop = min(stop_15m, stop_8h)
                src_sig = "+".join(reasons)
        elif br:
            entry = br["entry"]
            stop = br["stop"]
            src_sig = "15m_breakout"

        if entry and stop and atr > 0:
            # score: combine momentum and breakout strength
            score = 0.0
            if mom["pct_8h"] is not None:
                score += (mom["pct_8h"] / 100.0) * max(1.0, (mom["rvol_8h"] or 1.0))
            if mom["pct_24h"] is not None:
                score += 0.5 * (mom["pct_24h"] / 100.0)
            if br:
                score += 0.25 * br["rvol"] if br.get("rvol") else 0.0

            t1 = entry + 0.8 * atr
            t2 = entry + 1.5 * atr

            candidates.append({
                "ticker": tkr,
                "symbol": u["symbol"],
                "src": src,
                "signal": src_sig or "momentum",
                "entry": round(float(entry), 8),
                "stop": round(float(stop), 8),
                "t1": round(float(t1), 8),
                "t2": round(float(t2), 8),
                "atr": round(float(atr), 8),
                "pct_8h": mom["pct_8h"],
                "rvol_8h": mom["rvol_8h"],
                "usdvol_8h": mom["usdvol_8h"],
                "pct_24h": mom["pct_24h"],
                "score": round(float(score), 6)
            })

        time.sleep(PAUSE_SEC_PER_SYMBOL)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    snapshot["candidates"] = candidates[:10]

    # Output
    if blocked_451:
        snapshot["regime"] = {"ok": False, "reason": "binance-451-block"}

    if not candidates:
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "C", "text": "Hold and wait."})
        run_ms = int((perf_counter() - t0) * 1000)
        write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": _now_iso()})
        write_json(DEBUG_PATH, DEBUG_LOG)
        write_json(SUMMARY_PATH, {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regime": snapshot["regime"],
            "equity": equity,
            "cash": cash,
            "candidates": candidates,
            "signals": {"type": "C", "text": "Hold and wait."},
            "run_stats": {"elapsed_ms": run_ms, "time": _now_iso()}
        })
        print("Hold and wait. (No candidates.)")
        return

    # Select top and size with regime-aware cap (but do NOT block on regime)
    top = candidates[0]
    entry = float(top["entry"])
    stop = float(top["stop"])
    strong = bool(snapshot["regime"].get("ok", False))
    buy_usd = position_size(entry, stop, equity, cash, strong)

    plan_lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.8f}",
        f"• Target: T1 {top['t1']:.8f}, T2 {top['t2']:.8f}, trail after +1.0R",
        f"• Stop-loss: {stop:.8f} (breakout/8h low)",
        "• Portfolio rule: Exclude ETH, DOT; if cash needed, sell weakest on support break.",
        f"• USD buy size (risk 1.2%, cap {'60%' if strong else '30%'}): ${buy_usd:,.2f}",
        f"• Signal source: {top['signal']} via {top['src']}"
    ]

    write_json(SIGNAL_PATH, {"type": "B", "text": "\n".join(plan_lines),
                             "selected": {"ticker": top["ticker"], "entry": entry, "stop": stop,
                                          "t1": top["t1"], "t2": top["t2"], "atr": top["atr"],
                                          "src": top["src"], "signal": top["signal"], "buy_usd": buy_usd}})
    write_json(SNAP_PATH, snapshot)
    run_ms = int((perf_counter() - t0) * 1000)
    write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": _now_iso()})
    write_json(DEBUG_PATH, DEBUG_LOG)

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
                "t1": top["t1"],
                "t2": top["t2"],
                "atr": top["atr"],
                "src": top["src"],
                "signal": top["signal"],
                "buy_usd": buy_usd
            }
        },
        "run_stats": {"elapsed_ms": run_ms, "time": _now_iso()}
    })

    print("\n".join(plan_lines))


if __name__ == "__main__":
    main()