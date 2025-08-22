#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

# ------------------- paths & artifacts -------------------

ROOT = Path(__file__).resolve().parents[1]  # repository root
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True, parents=True)

SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNAL_PATH = DATA / "signals.json"
DEBUG_PATH = ARTIFACTS / "debug_scan.json"

# ------------------- small utils -------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sleep_ms(ms: int) -> None:
    time.sleep(ms / 1000.0)

DEBUG_LOG: List[Dict[str, Any]] = []

def _http_json(url: str, timeout: int = 20) -> Optional[Any]:
    """GET JSON with a UA header. Returns parsed JSON or None on error."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "rev-coinbase-scout/1.0", "Accept": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        ms = int((time.perf_counter() - t0) * 1000)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            data = None
        DEBUG_LOG.append({"url": url, "ok": True, "ms": ms})
        return data
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        DEBUG_LOG.append({"url": url, "ok": False, "ms": ms, "err": str(e)})
        return None

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------- TA helpers (no pandas) -------------------

def ema(values: List[float], span: int) -> List[float]:
    """Exponential moving average (EMA). Returns list same length as input."""
    if not values:
        return []
    if span < 1:
        return values[:]
    alpha = 2.0 / (span + 1.0)
    out: List[float] = []
    prev = values[0]
    out.append(prev)
    for v in values[1:]:
        prev = alpha * v + (1.0 - alpha) * prev
        out.append(prev)
    return out

def vwap_from_bars(bars: List[Dict[str, float]]) -> float:
    """VWAP using close * volume / sum(volume) on given bars list."""
    tot_pv = 0.0
    tot_v = 0.0
    for b in bars:
        p = float(b["c"])
        v = float(b["v"])
        tot_pv += p * v
        tot_v += v
    return (tot_pv / tot_v) if tot_v > 0 else 0.0

def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict[str, float]], period: int = 14) -> float:
    """ATR from typical candle list; assume ascending time order."""
    if len(bars) < period + 1:
        return 0.0
    trs: List[float] = []
    for i in range(1, period + 1):
        h = float(bars[-i]["h"])
        l = float(bars[-i]["l"])
        pc = float(bars[-i - 1]["c"])
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

def median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])

# ------------------- Coinbase public API -------------------
# Docs: https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproducts
CB_BASE = "https://api.exchange.coinbase.com"

def cb_products() -> List[Dict[str, Any]]:
    url = f"{CB_BASE}/products"
    data = _http_json(url)
    if isinstance(data, list):
        return data
    return []

def cb_stats(product_id: str) -> Optional[Dict[str, Any]]:
    # 24h stats: last/volume (base)
    url = f"{CB_BASE}/products/{urllib.parse.quote(product_id)}/stats"
    return _http_json(url)

def cb_book_l1(product_id: str) -> Optional[Dict[str, Any]]:
    # L1 best bid/ask
    url = f"{CB_BASE}/products/{urllib.parse.quote(product_id)}/book?level=1"
    return _http_json(url)

def cb_candles(product_id: str, granularity: int = 60, limit: int = 200) -> List[Dict[str, float]]:
    """
    Returns ascending list of bars:
      {"ts":sec,"o":...,"h":...,"l":...,"c":...,"v":...}
    Coinbase requires start & end; we'll set to now - limit*granularity .. now.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(seconds=granularity * limit + 30)
    params = urllib.parse.urlencode({
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "granularity": granularity
    })
    url = f"{CB_BASE}/products/{urllib.parse.quote(product_id)}/candles?{params}"
    arr = _http_json(url)
    # Candles come as arrays: [time, low, high, open, close, volume] likely newest->oldest
    out: List[Dict[str, float]] = []
    if isinstance(arr, list):
        try:
            # ensure ascending by time
            arr.sort(key=lambda x: x[0])
            for x in arr:
                out.append({
                    "ts": float(x[0]),
                    "o": float(x[3]),
                    "h": float(x[2]),
                    "l": float(x[1]),
                    "c": float(x[4]),
                    "v": float(x[5]),
                })
        except Exception:
            return []
    return out

# ------------------- Strategy logic -------------------

def check_regime() -> Dict[str, Any]:
    """Use BTC-USD 1m candles (last 120) on Coinbase: last > VWAP & last > EMA9."""
    bars = cb_candles("BTC-USD", granularity=60, limit=120)
    if not bars:
        return {"ok": False, "reason": "no-btc-1m"}

    closes = [b["c"] for b in bars]
    last = closes[-1]
    vwap_val = vwap_from_bars(bars[-60:])  # vwap recent hour
    ema9_val = ema(closes, span=9)[-1]

    ok = (last > vwap_val) and (last > ema9_val)
    return {
        "ok": ok,
        "reason": "" if ok else "btc-below-vwap-or-ema",
        "last": last,
        "vwap": vwap_val,
        "ema9": ema9_val,
    }

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid

def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Rule: 1-minute +1.8%..+4.0% vs previous close, RVOL >= 4 vs last 15 bars median volume,
    and closes above last 15-minute high.
    """
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]
    prev = bars_1m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    if pct < 0.018 or pct > 0.040:
        return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]])
    rvol = (last["v"] / vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last": last["c"]}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Require small fade & micro pullback in the last bar."""
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] <= 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "pullback_low": last["l"]}
    return None

def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    """
    1.2% risk, allocation cap 60% if strong regime else 30%, and respect cash.
    """
    risk_dollars = max(0.0, equity) * 0.012
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    alloc_cap = 0.60 if strong_regime else 0.30
    usd_cap = equity * alloc_cap
    return max(0.0, min(usd, usd_cap, cash))

# ------------------- Universe from Coinbase -------------------

def load_revolut_tickers() -> List[str]:
    """
    Load tickers from data/revolut_mapping.json (or fallback file list).
    We only need the ticker symbols; ignore other fields.
    """
    mapping_file = DATA / "revolut_mapping.json"
    if mapping_file.exists():
        try:
            arr = json.loads(mapping_file.read_text(encoding="utf-8"))
            tickers: List[str] = []
            for rec in arr:
                if isinstance(rec, dict):
                    t = (rec.get("ticker") or "").upper()
                    if t:
                        tickers.append(t)
            return tickers
        except Exception:
            pass
    # fallback: try revolut_list.txt (one ticker per line)
    lst = DATA / "revolut_list.txt"
    if lst.exists():
        return [line.strip().upper() for line in lst.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []

def coinbase_usd_universe(exclude: List[str]) -> List[Dict[str, str]]:
    """
    Cross-reference Revolut tickers to Coinbase USD products.
    Returns list of {ticker, product_id}.
    """
    rev_tickers = set(load_revolut_tickers())
    prods = cb_products()
    out: List[Dict[str, str]] = []
    for p in prods:
        try:
            pid = p["id"]  # e.g., "ARB-USD"
            base, quote = pid.split("-")
            if quote != "USD":
                continue
            t = base.upper()
            if t in rev_tickers and t not in exclude:
                out.append({"ticker": t, "product_id": pid})
        except Exception:
            continue
    return out

def product_filters(p: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    For a product {"ticker","product_id"}, fetch 24h stats + L1 book.
    Accept if 24h base_volume * last_price >= $8M and spread <= 0.5%.
    """
    pid = p["product_id"]
    st = cb_stats(pid) or {}
    last = float(st.get("last", 0.0) or 0.0)
    vol_base = float(st.get("volume", 0.0) or 0.0)
    vol_usd = last * vol_base

    ob = cb_book_l1(pid) or {}
    try:
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])
    except Exception:
        bid = 0.0
        ask = 0.0
    spr = spread_pct(bid, ask)

    if vol_usd >= 8_000_000 and spr <= 0.005 and last > 0:
        return {
            "ticker": p["ticker"],
            "product_id": pid,
            "price": last,
            "bid": bid,
            "ask": ask,
            "spread": spr,
            "vol_usd": vol_usd,
        }
    return None

# ------------------- Main orchestration -------------------

def main() -> None:
    # Portfolio inputs from env (with defaults matching your snapshot)
    equity = float(os.getenv("EQUITY", "41000") or "41000")
    cash   = float(os.getenv("CASH",   "32000") or "32000")

    # Safety floor / buffer logic
    floor = 36500.0
    buffer_ok = (equity - floor) >= 1000.0

    snapshot: Dict[str, Any] = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": [],
    }

    # Floor check: if breached, write & exit with “A” guidance
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps(
            {"type": "A", "text": "Raise cash now: exit weakest non-ETH, non-DOT on support breaks; halt new entries."},
            indent=2))
        print("Raise cash now: exit weakest non-ETH, non-DOT positions on support breaks; halt new entries.")
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        return

    # 1) Regime on BTC
    regime = check_regime()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # 2) Universe (exclude ETH, DOT from rotation)
    universe_raw = coinbase_usd_universe(exclude=["ETH", "DOT"])
    # Filter by stats & spread
    universe: List[Dict[str, Any]] = []
    for item in universe_raw:
        filt = product_filters(item)
        if filt:
            universe.append(filt)
        _sleep_ms(80)  # be polite

    snapshot["universe_count"] = len(universe)

    # 3) Signals: aggressive breakout + micro pullback
    scored: List[Dict[str, Any]] = []
    for u in universe:
        bars = cb_candles(u["product_id"], granularity=60, limit=120)
        if not bars:
            continue
        br = aggressive_breakout(bars)
        if not br:
            continue
        mp = micro_pullback_ok(bars)
        if not mp:
            continue

        atr1m = atr_from_klines(bars, period=14)
        if atr1m <= 0:
            continue

        score = br["rvol"] * (1.0 + br["pct"])
        cand = {
            "ticker": u["ticker"],
            "product": u["product_id"],
            "entry": mp["entry"],
            "pullback_low": mp["pullback_low"],
            "atr1m": atr1m,
            "rvol": br["rvol"],
            "pct": br["pct"],
            "score": score,
        }
        scored.append(cand)
        _sleep_ms(60)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # No candidates: write C and exit
    if not scored:
        snapshot["candidates"] = []
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type": "C", "text": "Hold and wait. (No qualified candidates.)"}, indent=2))
        print("Hold and wait. (No qualified candidates.)")
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        return

    # Take the single best candidate
    top = scored[0]
    entry = float(top["entry"])
    stop  = float(top["pullback_low"])
    atr   = float(top["atr1m"])

    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr

    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    snapshot["candidates"] = [top]
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))

    # Build “B” style output
    sell_plan = "Sell weakest non-ETH, non-DOT on support break if cash needed."
    signal_text = "\n".join([
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min",
        f"• What to sell from portfolio (excluding ETH, DOT): {sell_plan}",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}",
    ])
    SIGNAL_PATH.write_text(json.dumps({"type": "B", "text": signal_text}, indent=2))

    print(signal_text)
    DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))


if __name__ == "__main__":
    main()