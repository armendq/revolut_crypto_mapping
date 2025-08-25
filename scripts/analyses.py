#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# -------- Paths
OUT_DIR = Path("public_runs/latest")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.json"
SNAPSHOT = OUT_DIR / "market_snapshot.json"
DEBUG = OUT_DIR / "debug_scan.json"
RUNSTATS = OUT_DIR / "run_stats.json"
SIGNALS = OUT_DIR / "signals.json"
DATA = Path("data")
DATA.mkdir(exist_ok=True)

# -------- Risk/alloc
EQUITY = float(os.getenv("EQUITY", "41000") or "41000")
CASH   = float(os.getenv("CASH", "32000") or "32000")
EQUITY_FLOOR = 40000.0
RISK_PCT = 0.012
CAP_STRONG = 0.60
CAP_WEAK = 0.30

# -------- Binance mirrors and endpoints
BINANCE_BASES = [
    "https://api4.binance.com",
    "https://api-gcp.binance.com",
    "https://data-api.binance.vision",
]
KLINES_PATH = "/api/v3/klines"
TICKER24_PATH = "/api/v3/ticker/24hr"
DEPTH_PATH = "/api/v3/depth"

# -------- Fetch settings
RETRIES = 3
TIMEOUT = 20
SLEEP_BASE = 0.9

# -------- Filters and logic
MIN_QUOTE_VOL_24H = 3_000_000
MAX_SPREAD = 0.006
UNROTATE = {"ETH", "DOT"}
UNTOUCHABLE = {"DOT"}

ATR_LEN_5M = 14
EMA_LEN = 20
BREAKOUT_LOOKBACK_5M = 20
RVOL_WINDOW_5M = 20
RVOL_MIN = 3.0
BAR_RET_MIN = 0.012
BAR_RET_MAX = 0.06
HTF_CONFIRM_EMA = 20
HTF_CONFIRM = True

VERBOSE = True
def log(*a: Any) -> None:
    if VERBOSE:
        print("[analyses]", *a, flush=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

blocked_451 = False

# -------- HTTP with mirror rotation
def _get_json_across_bases(path: str, params: Dict[str, Any]) -> Optional[Any]:
    global blocked_451
    last_err = None
    for base in BINANCE_BASES:
        url = f"{base}{path}"
        for attempt in range(1, RETRIES + 1):
            try:
                r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent":"rev-scan/1.2"})
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 451:
                    blocked_451 = True
                last_err = f"http {r.status_code}"
            except Exception as e:
                last_err = f"exc {e}"
            time.sleep(SLEEP_BASE * attempt)
    if last_err:
        log("GET fail:", path, last_err)
    return None

# -------- Universe (from mapping or 24h tickers)
def get_universe() -> List[str]:
    mapping = DATA / "revolut_mapping.json"
    if mapping.exists():
        try:
            obj = json.loads(mapping.read_text(encoding="utf-8"))
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
            return sorted(set(out))
        except Exception as e:
            log("mapping parse error:", e)
    # fallback: top USDT by quote volume
    data = _get_json_across_bases(TICKER24_PATH, {})
    if not isinstance(data, list):
        return []
    rows = []
    for d in data:
        s = d.get("symbol", "")
        if not s.endswith("USDT"): continue
        if any(x in s for x in ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT")): continue
        qv = float(d.get("quoteVolume", "0") or 0)
        rows.append((s[:-4], qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [t for t,_ in rows[:200]]

# -------- Data fetchers
def fetch_klines(sym: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    arr = _get_json_across_bases(KLINES_PATH, {"symbol": f"{sym}USDT", "interval": interval, "limit": limit})
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

def fetch_orderbook(sym: str) -> Optional[Tuple[float,float]]:
    d = _get_json_across_bases(DEPTH_PATH, {"symbol": f"{sym}USDT", "limit": 5})
    if not d or "bids" not in d or "asks" not in d:
        return None
    try:
        bid = float(d["bids"][0][0]); ask = float(d["asks"][0][0])
        return bid, ask
    except Exception:
        return None

def fetch_24h(sym: str) -> Optional[Dict[str,float]]:
    d = _get_json_across_bases(TICKER24_PATH, {"symbol": f"{sym}USDT"})
    if not d or isinstance(d, list):
        return None
    try:
        return {"last": float(d["lastPrice"]), "quote_vol": float(d["quoteVolume"])}
    except Exception:
        return None

# -------- TA
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
    if len(df) < window + 1: return 0.0
    med = df["v"].iloc[-window-1:-1].median()
    return 0.0 if med <= 0 else float(df["v"].iloc[-1] / med)

# -------- Sizing
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

# -------- Regime
def btc_regime() -> Dict[str, Any]:
    k = fetch_klines("BTC", "5m", 288)
    if k is None or len(k) < 60:
        return {"ok": False, "reason": "no_btc_5m"}
    k = k.rename(columns={"o":"o","h":"h","l":"l","c":"c","v":"v"})
    e9 = ema(k["c"], 9).iloc[-1]
    vw = vwap(k).iloc[-1]
    last = k["c"].iloc[-1]
    ok = (last > e9) and (last > vw)
    return {"ok": bool(ok), "last": float(last), "ema9": float(e9), "vwap": float(vw)}

# -------- Per-symbol eval
def eval_symbol(sym: str, dbg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stats24 = fetch_24h(sym)
    if not stats24 or stats24["quote_vol"] < MIN_QUOTE_VOL_24H:
        dbg.setdefault("skip", []).append({"sym": sym, "reason": "low_liquidity_or_24h"})
        return None
    ob = fetch_orderbook(sym)
    if not ob:
        dbg.setdefault("skip", []).append({"sym": sym, "reason": "no_orderbook"})
        return None
    bid, ask = ob
    spr = spread_pct(bid, ask)
    if spr > MAX_SPREAD:
        dbg.setdefault("skip", []).append({"sym": sym, "reason": "wide_spread", "spread": spr})
        return None

    k5 = fetch_klines(sym, "5m", 240)
    k1h = fetch_klines(sym, "1h", 300)
    if k5 is None or k1h is None or len(k5) < (ATR_LEN_5M + BREAKOUT_LOOKBACK_5M + 5) or len(k1h) < (HTF_CONFIRM_EMA + 5):
        dbg.setdefault("skip", []).append({"sym": sym, "reason": "insufficient_bars"})
        return None

    k5 = k5.rename(columns={"o":"o","h":"h","l":"l","c":"c","v":"v"})
    k1h = k1h.rename(columns={"o":"o","h":"h","l":"l","c":"c","v":"v"})
    k5["ema"] = ema(k5["c"], EMA_LEN)
    k5["atr"] = atr(k5, ATR_LEN_5M)
    k5["vwap"] = vwap(k5)
    k1h["ema"] = ema(k1h["c"], HTF_CONFIRM_EMA)
    k1h["vwap"] = vwap(k1h)

    last5 = k5.index[-1]
    prev5 = k5.index[-2]
    hh = float(k5["h"].iloc[-BREAKOUT_LOOKBACK_5M-1:-1].max())
    close = float(k5.at[last5, "c"])
    prev_close = float(k5.at[prev5, "c"])
    bar_ret = (close / prev_close) - 1.0
    rvol = rel_volume(k5, RVOL_WINDOW_5M)

    if not (close > hh and RVOL_MIN <= rvol and BAR_RET_MIN <= bar_ret <= BAR_RET_MAX):
        dbg.setdefault("skip", []).append({"sym": sym, "reason": "no_5m_breakout", "rvol": rvol, "bar_ret": bar_ret})
        return None

    if HTF_CONFIRM:
        h_ok = (k1h["c"].iloc[-1] > k1h["ema"].iloc[-1]) and (k1h["c"].iloc[-1] > k1h["vwap"].iloc[-1])
        if not h_ok:
            dbg.setdefault("skip", []).append({"sym": sym, "reason": "1h_not_confirmed"})
            return None

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

# -------- Main
def main():
    t0 = time.time()
    events: List[Any] = []
    guard = (EQUITY - EQUITY_FLOOR) < 1000.0 or EQUITY <= (EQUITY_FLOOR + 500.0)
    if guard:
        sig = {"type": "A", "text": "Raise cash now: halt entries; sell weakest on breaks."}
        out = {
            "status": "ok",
            "ts_utc": now_iso(),
            "regime": {"ok": False, "reason": "equity_floor_guard"},
            "equity": EQUITY, "cash": CASH,
            "candidates": [],
            "signals": sig
        }
        SUMMARY.write_text(json.dumps(out, indent=2))
        SIGNALS.write_text(json.dumps(sig, indent=2))
        RUNSTATS.write_text(json.dumps({"duration_sec": round(time.time()-t0,2), "wrote": "guard"}, indent=2))
        SNAPSHOT.write_text(json.dumps({"note":"guard_triggered"}, indent=2))
        DEBUG.write_text(json.dumps({"events": events}, indent=2))
        print("[analyses] guard fired; wrote raise-cash signal")
        return

    uni = get_universe()
    events.append({"universe_count": len(uni)})
    log("universe:", len(uni))

    regime = btc_regime()
    strong = bool(regime.get("ok", False))
    events.append({"regime": regime})

    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for i, sym in enumerate(uni, 1):
        if sym in UNTOUCHABLE:
            skipped.append({"symbol": sym, "reason": "untouchable"})
            continue
        # quick probe for binance spot availability
        probe = _get_json_across_bases(TICKER24_PATH, {"symbol": f"{sym}USDT"})
        if probe is None or isinstance(probe, list) or "lastPrice" not in probe:
            skipped.append({"symbol": sym, "reason": "no_binance_spot"})
            continue
        try:
            c = eval_symbol(sym, {"skip": []})
            if c:
                candidates.append(c)
        except Exception as e:
            skipped.append({"symbol": sym, "reason": f"exception:{type(e).__name__}"})
        if i % 25 == 0:
            log(f"scanned {i}/{len(uni)}")

    def score(x: Dict[str, Any]) -> float:
        r = x.get("rvol5m", 0.0)
        spr = x.get("spread", 1.0)
        qv = math.log10(max(1.0, x.get("quote_vol_24h", 1.0)))
        br = x.get("bar_ret", 0.0)
        br_bonus = -abs(br - 0.03) * 10
        return 3.0*r + 1.5*qv + br_bonus - 5.0*spr

    candidates.sort(key=score, reverse=True)

    # 451 handling
    if blocked_451 and not candidates:
        regime = {"ok": False, "reason": "binance-451-block"}
        strong = False

    if candidates and strong:
        top = candidates[0]
        entry = float(top["entry"]); stop = float(top["stop"])
        buy_usd = position_usd(entry, stop, EQUITY, CASH, strong)
        signals = {
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
    elif candidates:
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
        txt = "Hold and wait."
        if blocked_451:
            txt = "Hold and wait. Binance blocked (451)."
        signals = {"type": "C", "text": txt}

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
        "universe_count": len(uni),
        "scanned": len(uni) - len(UNTOUCHABLE),
        "skipped_count": len(skipped),
        "skipped": skipped[:100]
    }
    runstats = {
        "duration_sec": round(time.time()-t0, 2),
        "time_utc": now_iso(),
        "blocked_451": blocked_451
    }

    SUMMARY.write_text(json.dumps(summary, indent=2))
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2))
    DEBUG.write_text(json.dumps({"events": [{"blocked_451": blocked_451}]}, indent=2))
    RUNSTATS.write_text(json.dumps(runstats, indent=2))
    SIGNALS.write_text(json.dumps(signals, indent=2))

    print(f"[analyses] wrote summary with {len(candidates)} candidates in {runstats['duration_sec']}s", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fail = {
            "status": "error",
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "error": "fatal",
            "signals": {"type":"C","text":"Hold and wait. Exception."}
        }
        SUMMARY.write_text(json.dumps(fail, indent=2))
        SNAPSHOT.write_text(json.dumps({"fatal": True}, indent=2))
        DEBUG.write_text(json.dumps({"traceback": traceback.format_exc()}, indent=2))
        RUNSTATS.write_text(json.dumps({"duration_sec": None, "time_utc": now_iso(), "fatal": True}, indent=2))
        print("[analyses] FAILED but wrote minimal outputs", flush=True)
        raise