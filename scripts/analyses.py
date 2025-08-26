#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, time, pathlib, statistics
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from time import perf_counter

import requests
import pandas as pd

from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------------- Paths ----------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
PUB_LATEST = ROOT / "public_runs" / "latest"
for p in (ARTIFACTS, DATA, PUB_LATEST):
    p.mkdir(parents=True, exist_ok=True)

SNAP_PATH    = ARTIFACTS / "market_snapshot.json"
SIGNAL_PATH  = DATA / "signals.json"
RUN_STATS    = ARTIFACTS / "run_stats.json"
DEBUG_PATH   = ARTIFACTS / "debug_scan.json"
SUMMARY_PATH = PUB_LATEST / "summary.json"

DEBUG: List[Dict[str, Any]] = []

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(event: str, **fields):
    DEBUG.append({"t": now_iso(), "event": event, **fields})

def write_json(path: pathlib.Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def as_float(x: Any) -> float:
    if hasattr(x, "iloc"):
        return float(x.iloc[-1])
    return float(x)

# --------------- Config ---------------
# Floor
EQUITY_FLOOR = float(os.getenv("EQUITY_FLOOR", "40000"))

# Risk/sizing
RISK_PCT      = float(os.getenv("RISK_PCT", "0.012"))
STRONG_ALLOC  = float(os.getenv("STRONG_ALLOC", "0.60"))
WEAK_ALLOC    = float(os.getenv("WEAK_ALLOC",  "0.30"))

# Liquidity/quality gates (env-tunable)
VOL_USD_MIN_24H = float(os.getenv("VOL_USD_MIN_24H", "8000000"))
MAX_SPREAD      = float(os.getenv("MAX_SPREAD", "0.005"))

# Momentum gates
WIN_H        = 8
WIN_24H      = 24
MIN_PCT_8H   = float(os.getenv("MIN_PCT_8H", "12"))
MIN_RVOL_8H  = float(os.getenv("MIN_RVOL_8H", "3"))
MIN_VOL_8H_USD = float(os.getenv("MIN_VOL_8H_USD", "1500000"))
MIN_PCT_24H  = float(os.getenv("MIN_PCT_24H", "20"))
ATR_LEN      = 14

# Exclusions
EXCLUDE_ROTATION = {"ETH","DOT"}  # DOT staked; ETH/DOT not rotated

# Throttle
PAUSE_SEC_PER_SYMBOL = float(os.getenv("PAUSE_SEC_PER_SYMBOL", "0.02"))

# --------------- HTTP -----------------
BINANCE_BASE = "https://data-api.binance.vision"
COINBASE_BASE = "https://api.exchange.coinbase.com"
H_TIMEOUT = 20
RETRIES   = 4
BACKOFF   = 1.6
blocked_451 = False

S = requests.Session()
S.headers.update({
    "User-Agent": "rev-analyses/3.1 (+https://github.com/armendq/revolut_crypto_mapping)",
    "Accept": "application/json,text/plain,*/*",
})

def http_get_json(url: str, params: Optional[Dict[str, Any]]=None) -> Any:
    global blocked_451
    last_err = None
    for i in range(1, RETRIES+1):
        try:
            r = S.get(url, params=params, timeout=H_TIMEOUT)
            if r.status_code == 200:
                log("http_ok", url=url, params=params or {}, try_num=i)
                return r.json()
            if r.status_code == 451:
                blocked_451 = True
            last_err = f"HTTP {r.status_code}"
            log("http_non200", url=url, status=r.status_code, try_num=i, body=r.text[:200])
        except Exception as e:
            last_err = str(e)
            log("http_exc", url=url, err=last_err, try_num=i)
        time.sleep(BACKOFF ** (i-1))
    raise RuntimeError(f"GET failed: {url} err={last_err}")

# -------- Binance helpers (symbol required) --------
def bn_ticker_24h(symbol: str) -> Optional[Dict[str, float]]:
    try:
        d = http_get_json(f"{BINANCE_BASE}/api/v3/ticker/24hr", {"symbol": symbol})
        last = float(d.get("lastPrice", 0.0))
        vol_base = float(d.get("volume", 0.0))
        bid = float(d.get("bidPrice", 0.0))
        ask = float(d.get("askPrice", 0.0))
        mid = (bid+ask)/2.0 if bid>0 and ask>0 else 0.0
        spread = (ask-bid)/mid if mid>0 else 9.99
        return {"price": last, "vol_usd": vol_base*last, "bid": bid, "ask": ask, "spread": spread}
    except Exception as e:
        log("bn_24h_fail", symbol=symbol, err=str(e))
        return None

def bn_klines(symbol: str, interval: str, limit: int) -> List[List[Any]]:
    return http_get_json(f"{BINANCE_BASE}/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})

# -------- Coinbase helpers --------
def cb_product(ticker: str) -> str:
    return f"{ticker.upper()}-USD"

def cb_book_best(ticker: str) -> Optional[Dict[str, float]]:
    try:
        pid = cb_product(ticker)
        d = http_get_json(f"{COINBASE_BASE}/products/{pid}/book", {"level": 1})
        bid = float(d["bids"][0][0]); ask = float(d["asks"][0][0])
        mid = (bid+ask)/2.0 if bid>0 and ask>0 else 0.0
        spread = (ask-bid)/mid if mid>0 else 9.99
        return {"bid": bid, "ask": ask, "spread": spread}
    except Exception as e:
        log("cb_book_fail", ticker=ticker, err=str(e)); return None

def cb_stats_24h(ticker: str) -> Optional[Dict[str, float]]:
    try:
        pid = cb_product(ticker)
        d = http_get_json(f"{COINBASE_BASE}/products/{pid}/stats")
        last = float(d.get("last", 0.0)); vol_base = float(d.get("volume", 0.0))
        return {"price": last, "vol_usd": vol_base*last}
    except Exception as e:
        log("cb_stats_fail", ticker=ticker, err=str(e)); return None

def cb_candles(ticker: str, granularity: int, limit: int) -> List[List[Any]]:
    pid = cb_product(ticker)
    arr = http_get_json(f"{COINBASE_BASE}/products/{pid}/candles", {"granularity": granularity})
    if not isinstance(arr, list): return []
    arr.sort(key=lambda r: r[0])
    return arr[-limit:]

# --------------- TA -------------------
def true_range(h: float, l: float, pc: float) -> float:
    return max(h-l, abs(h-pc), abs(l-pc))

def atr_from_ohlcv(bars: List[Dict[str, float]], length: int) -> float:
    if len(bars) < length+1: return 0.0
    trs = []
    for i in range(1, length+1):
        pc = bars[-i-1]["c"]; hi = bars[-i]["h"]; lo = bars[-i]["l"]
        trs.append(true_range(hi, lo, pc))
    return statistics.fmean(trs) if trs else 0.0

def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

# ------------- Momentum/Signals --------------
def compute_momentum_1h(kl: List[List[Any]]) -> Dict[str, Any]:
    out = {"pct_8h": None, "rvol_8h": None, "usdvol_8h": None, "pct_24h": None, "last_low_8h": None}
    if not kl or len(kl) < 26: return out
    closes = [float(r[4]) for r in kl]
    lows   = [float(r[3]) for r in kl]
    vols   = [float(r[5]) for r in kl]
    last = closes[-1]

    ref8 = closes[-(WIN_H+1)]; pct8 = (last-ref8)/ref8*100 if ref8>0 else 0.0
    last8 = sum(vols[-WIN_H:]); prev8 = sum(vols[-2*WIN_H:-WIN_H]) or 1e-9
    rvol8 = last8/prev8
    avg8 = statistics.fmean(closes[-WIN_H:])
    usd8 = avg8*last8

    ref24 = closes[-(WIN_24H+1)]; pct24 = (last-ref24)/ref24*100 if ref24>0 else 0.0

    out.update({"pct_8h": pct8, "rvol_8h": rvol8, "usdvol_8h": usd8, "pct_24h": pct24, "last_low_8h": min(lows[-WIN_H:])})
    return out

def breakout_15m(bars: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars) < 20: return None
    last, prev = bars[-1], bars[-2]
    pct = (last["c"]/prev["c"]) - 1.0
    if pct < 0.008 or pct > 0.05: return None
    vol_med = median([b["v"] for b in bars[-16:-1]])
    rvol = (last["v"]/vol_med) if vol_med>0 else 0.0
    if rvol < 2.5: return None
    hh15 = max(b["h"] for b in bars[-15:])
    if last["c"] <= hh15*1.0005: return None
    return {"pct": pct, "rvol": rvol, "entry": last["c"], "stop": last["l"]}

# --------------- Regime ----------------
def check_regime() -> Dict[str, Any]:
    bars = get_btc_5m_klines()
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        return {"ok": False, "reason": "no-btc-5m"}
    close = bars["close"]
    try:
        last = as_float(close.iloc[-1]); vw = as_float(vwap(bars)); e9 = as_float(ema(close, span=9))
    except Exception:
        return {"ok": False, "reason": "calc-error"}
    ok = (last > vw) and (last > e9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-or-ema", "last": last, "vwap": vw, "ema9": e9}

# -------------- Universe ----------------
def load_revolut_mapping() -> List[Dict[str, Any]]:
    path = DATA / "revolut_mapping.json"
    if not path.exists():
        log("mapping_missing", path=str(path))
        return []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for m in obj:
            if isinstance(m, dict): out.append(m)
            elif isinstance(m, str): out.append({"ticker": m})
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                vv = dict(v); vv.setdefault("ticker", k); out.append(vv)
            else:
                out.append({"ticker": k})
    log("mapping_loaded", count=len(out))
    return out

def best_binance_symbol(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("binance_symbol"): return str(entry["binance_symbol"]).upper()
    if entry.get("binance"):        return str(entry["binance"]).upper()
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# --------------- Sizing ----------------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = STRONG_ALLOC if strong_regime else WEAK_ALLOC
    usd_cap = equity * cap
    return max(0.0, min(usd, usd_cap, cash))

# --------------- Main ------------------
def main():
    t0 = perf_counter()
    log("run_start")

    equity = float(os.getenv("EQUITY", "41000") or "41000")
    cash   = float(os.getenv("CASH",   "32000") or "32000")

    snapshot = {
        "time": now_iso(),
        "equity": equity,
        "cash": cash,
        "floor": EQUITY_FLOOR,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": []
    }

    # Guard
    buffer_ok = (equity - EQUITY_FLOOR) >= 1000.0
    if not buffer_ok or equity <= (EQUITY_FLOOR + 500.0):
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity_near_floor"
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."})
        run_ms = int((perf_counter()-t0)*1000)
        write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": now_iso()})
        write_json(DEBUG_PATH, DEBUG)
        write_json(SUMMARY_PATH, {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "equity": equity, "cash": cash,
            "regime": {"ok": False, "reason": "capital-guard"},
            "candidates": [], "signals": {"type": "A", "text": "Raise cash now."},
            "run_stats": {"elapsed_ms": run_ms, "time": now_iso()}
        })
        print("Raise cash now.")
        return

    # Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong = bool(regime.get("ok", False))
    log("regime", **regime)

    # Mapping
    mapping = load_revolut_mapping()
    if len(mapping) == 0:
        # Hard fail so CI shows red when we’d otherwise exit in 1s
        run_ms = int((perf_counter()-t0)*1000)
        write_json(DEBUG_PATH, DEBUG)
        raise SystemExit("FATAL: revolut_mapping.json empty or missing")

    # Universe build
    universe = []
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if not tkr or tkr in EXCLUDE_ROTATION:
            continue

        sym = best_binance_symbol(m)
        meta = bn_ticker_24h(sym) if sym else None

        if meta and meta["vol_usd"] >= VOL_USD_MIN_24H and meta["spread"] <= MAX_SPREAD:
            universe.append({"ticker": tkr, "src": "binance", "symbol": sym, **meta})
        else:
            stats = cb_stats_24h(tkr)
            book  = cb_book_best(tkr)
            if stats and book and stats["vol_usd"] >= VOL_USD_MIN_24H and book["spread"] <= MAX_SPREAD:
                universe.append({"ticker": tkr, "src": "coinbase", "symbol": cb_product(tkr),
                                 "price": stats["price"], "vol_usd": stats["vol_usd"],
                                 "bid": book["bid"], "ask": book["ask"], "spread": book["spread"]})
        time.sleep(PAUSE_SEC_PER_SYMBOL)

    snapshot["universe_count"] = len(universe)
    log("universe_built", count=len(universe))

    if len(universe) == 0:
        # Hard fail to surface the problem instead of a 1s “Hold”
        write_json(SNAP_PATH, snapshot)
        write_json(DEBUG_PATH, DEBUG)
        raise SystemExit("FATAL: Universe empty after liquidity/spread filters")

    # Candidates
    candidates: List[Dict[str, Any]] = []
    for u in universe:
        tkr, src = u["ticker"], u["src"]

        # 1h klines
        if src == "binance":
            try: kl1h = bn_klines(u["symbol"], "1h", 120)
            except Exception as e: log("kl1h_bn_fail", symbol=u["symbol"], err=str(e)); kl1h=[]
        else:
            try: kl1h = cb_candles(tkr, 3600, 120)
            except Exception as e: log("kl1h_cb_fail", ticker=tkr, err=str(e)); kl1h=[]

        # 15m klines
        if src == "binance":
            try: kl15 = bn_klines(u["symbol"], "15m", 120)
            except Exception as e: log("kl15_bn_fail", symbol=u["symbol"], err=str(e)); kl15=[]
        else:
            try: kl15 = cb_candles(tkr, 900, 120)
            except Exception as e: log("kl15_cb_fail", ticker=tkr, err=str(e)); kl15=[]

        bars_15m: List[Dict[str, float]] = []
        for r in kl15:
            if len(r) < 6: continue
            if src == "binance":
                bars_15m.append({"ts": r[0], "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4]), "v": float(r[5])})
            else:
                bars_15m.append({"ts": r[0]*1000, "o": float(r[3]), "h": float(r[2]), "l": float(r[1]), "c": float(r[4]), "v": float(r[5])})

        mom = compute_momentum_1h(kl1h)
        mom_pass = False; reasons = []

        if mom["pct_8h"] is not None:
            if (mom["pct_8h"] >= MIN_PCT_8H and
                (mom["rvol_8h"] or 0.0) >= MIN_RVOL_8H and
                (mom["usdvol_8h"] or 0.0) >= MIN_VOL_8H_USD):
                mom_pass = True; reasons.append("8h_momentum")
        if not mom_pass and mom["pct_24h"] is not None and mom["pct_24h"] >= MIN_PCT_24H:
            mom_pass = True; reasons.append("24h_momentum")

        br = breakout_15m(bars_15m) if bars_15m else None
        atr = atr_from_ohlcv(bars_15m, ATR_LEN) if bars_15m else 0.0

        entry = stop = None; src_sig = None
        if mom_pass and bars_15m:
            last_c = bars_15m[-1]["c"]; stop_15m = bars_15m[-1]["l"]; stop_8h = mom.get("last_low_8h", stop_15m)
            entry = last_c; stop = min(stop_15m, stop_8h); src_sig = "+".join(reasons)
        elif br:
            entry = br["entry"]; stop = br["stop"]; src_sig = "15m_breakout"

        if entry and stop and atr > 0:
            score = 0.0
            if mom["pct_8h"] is not None:  score += (mom["pct_8h"]/100.0) * max(1.0, (mom["rvol_8h"] or 1.0))
            if mom["pct_24h"] is not None: score += 0.5 * (mom["pct_24h"]/100.0)
            if br and br.get("rvol"):      score += 0.25 * br["rvol"]

            t1 = entry + 0.8*atr; t2 = entry + 1.5*atr
            candidates.append({
                "ticker": tkr, "symbol": u["symbol"], "src": src, "signal": src_sig or "momentum",
                "entry": round(float(entry), 8), "stop": round(float(stop), 8),
                "t1": round(float(t1), 8), "t2": round(float(t2), 8),
                "atr": round(float(atr), 8),
                "pct_8h": mom["pct_8h"], "rvol_8h": mom["rvol_8h"], "usdvol_8h": mom["usdvol_8h"], "pct_24h": mom["pct_24h"]
            })

        time.sleep(PAUSE_SEC_PER_SYMBOL)

    candidates.sort(key=lambda x: (
        (x["pct_8h"] or 0)/100.0
        + 0.5*((x["pct_24h"] or 0)/100.0)
    ), reverse=True)

    snapshot["candidates"] = candidates[:10]
    if blocked_451:
        snapshot["regime"] = {"ok": False, "reason": "binance-451-block"}

    # No candidates -> explicit output
    if not candidates:
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "C", "text": "Hold and wait."})
        run_ms = int((perf_counter()-t0)*1000)
        write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": now_iso()})
        write_json(DEBUG_PATH, DEBUG)
        write_json(SUMMARY_PATH, {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regime": snapshot["regime"], "equity": equity, "cash": cash,
            "candidates": candidates, "signals": {"type": "C", "text": "Hold and wait."},
            "run_stats": {"elapsed_ms": run_ms, "time": now_iso()}
        })
        print("Hold and wait. (No candidates.)"); return

    # Size top pick
    top = candidates[0]
    entry = float(top["entry"]); stop = float(top["stop"])
    buy_usd = position_size(entry, stop, equity, cash, bool(snapshot["regime"].get("ok", False)))

    plan_lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.8f}",
        f"• Target: T1 {top['t1']:.8f}, T2 {top['t2']:.8f}, trail after +1.0R",
        f"• Stop-loss: {stop:.8f}",
        "• Portfolio: Exclude ETH, DOT; sell weakest on support break if cash needed.",
        f"• USD buy size: ${buy_usd:,.2f} (risk {RISK_PCT*100:.1f}%, cap {'60%' if snapshot['regime'].get('ok', False) else '30%'})",
        f"• Signal: {top['signal']} via {top['src']}"
    ]

    write_json(SIGNAL_PATH, {"type": "B", "text": "\n".join(plan_lines),
                             "selected": {**top, "buy_usd": buy_usd}})
    write_json(SNAP_PATH, snapshot)
    run_ms = int((perf_counter()-t0)*1000)
    write_json(RUN_STATS, {"elapsed_ms": run_ms, "time": now_iso()})
    write_json(DEBUG_PATH, DEBUG)
    write_json(SUMMARY_PATH, {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": snapshot["regime"], "equity": equity, "cash": cash,
        "candidates": candidates,
        "signals": {"type": "B", "text": "\n".join(plan_lines), "selected": {**top, "buy_usd": buy_usd}},
        "run_stats": {"elapsed_ms": run_ms, "time": now_iso()}
    })
    print("\n".join(plan_lines))

if __name__ == "__main__":
    main()