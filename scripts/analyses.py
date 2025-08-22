#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses runner:
- Loads Revolut mapping (data/revolut_mapping.json)
- Pulls lightweight data from public Binance endpoints
- Checks regime (BTC 5m close > VWAP & > 9-EMA)
- Scans for Aggressive Breakout + Micro Pullback on 1m bars
- Writes:
    data/market_snapshot.json
    data/signals.json
The files are always refreshed (new ISO time) so the workflow can commit.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import urllib.request

# -------------------- constants / paths --------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SNAP_PATH = DATA_DIR / "market_snapshot.json"
SIGNALS_PATH = DATA_DIR / "signals.json"

FLOOR = 36_500.0
BUFFER_MIN = 1_000.0

# -------------------- small utils --------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _json_get(url: str, timeout: int = 15) -> Optional[dict | list]:
    req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def write_snapshot(payload: dict) -> None:
    # Always inject fresh timestamp so a commit is produced even if nothing else changed
    payload = dict(payload)
    payload["time"] = _now_iso()
    SNAP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def write_signal(text: str, type_code: str) -> None:
    SIGNALS_PATH.write_text(
        json.dumps({"type": type_code, "text": text, "time": _now_iso()}, indent=2),
        encoding="utf-8",
    )

# -------------------- minimal TA helpers --------------------

def ema_series(series: pd.Series, span: int) -> pd.Series:
    # pandas ewm already returns a Series; we only need the last value most of the time
    return series.ewm(span=span, adjust=False).mean()

def vwap_df(df: pd.DataFrame) -> float:
    """
    df columns: open, high, low, close, volume
    Returns the *last* rolling VWAP (single float) for the whole frame.
    """
    # Typical price
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.cumsum()
    cum_vol = df["volume"].cumsum().replace(0, 1e-12)
    vwap_series = cum_pv / cum_vol
    return float(vwap_series.iloc[-1])

# -------------------- Binance lightweight adapters --------------------

def binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Returns pandas DataFrame with columns [open, high, low, close, volume] (floats).
    If anything fails, returns empty DataFrame.
    """
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api3.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    ]
    for u in urls:
        try:
            raw = _json_get(u)
            if not raw:
                continue
            df = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","trades","tbbav","tbqav","ignore"
            ])
            # Convert necessary cols to float
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            return df[["open","high","low","close","volume"]].copy()
        except Exception:
            continue
    return pd.DataFrame(columns=["open","high","low","close","volume"])

def binance_24h_and_book(symbol: str) -> dict:
    """
    Returns dict {'ok', 'price', 'volume_usd', 'bid', 'ask'} for SYMBOL/USDT pairs.
    """
    try:
        t = _json_get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")
        if not t:
            return {"ok": False}
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last
        ob = _json_get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=5")
        bid = float(ob["bids"][0][0]); ask = float(ob["asks"][0][0])
        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
    except Exception:
        return {"ok": False}

# -------------------- regime --------------------

def check_regime_btc_5m() -> dict:
    """
    Longs only if BTCUSDT 5m close > 5m VWAP AND > 9-EMA.
    Returns {'ok': bool, 'reason': str}
    """
    df = binance_klines("BTCUSDT", "5m", limit=200)
    if df.empty:
        return {"ok": False, "reason": "no-binance-klines"}

    close = df["close"]
    vw = vwap_df(df)                        # float
    ema9_last = float(ema_series(close, 9).iloc[-1])
    last_close = float(close.iloc[-1])

    ok = (last_close > vw) and (last_close > ema9_last)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-ema"}

# -------------------- breakout logic --------------------

@dataclass
class KBar:
    o: float; h: float; l: float; c: float; v: float

def one_minute_bars(symbol: str, limit: int = 120) -> List[KBar]:
    df = binance_klines(symbol, "1m", limit=limit)
    if df.empty:
        return []
    return [KBar(o=float(r.open), h=float(r.high), l=float(r.low), c=float(r.close), v=float(r.volume))
            for r in df.itertuples(index=False)]

def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float((s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0))

def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_1m(bars: List[KBar], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i].h; l = bars[-i].l; pc = bars[-i - 1].c
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

def aggressive_breakout(bars: List[KBar]) -> Optional[dict]:
    # Rule 3
    if len(bars) < 20:
        return None
    last, prev = bars[-1], bars[-2]
    pct = (last.c / prev.c) - 1.0
    if not (0.018 <= pct <= 0.04):
        return None
    vol_med = median([b.v for b in bars[-16:-1]])
    rvol = (last.v / vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b.h for b in bars[-15:])
    if last.c <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last": last}

def micro_pullback_ok(bars: List[KBar]) -> Optional[dict]:
    # Rule 4 (approximation without ticks)
    if len(bars) < 2:
        return None
    last = bars[-1]
    if last.h <= 0 or last.c <= 0:
        return None
    fade = (last.h - last.c) / last.h
    micro = (last.h - last.l) / last.h
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last.c, "pullback_low": last.l}
    return None

# -------------------- universe / mapping --------------------

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid

def load_revolut_mapping() -> List[dict]:
    with open(DATA_DIR / "revolut_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)

def symbol_on_binance(entry: dict) -> Optional[str]:
    # prefer explicit if mapping carries it, else ticker + USDT
    if entry.get("binance"):
        return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    if not t:
        return None
    return f"{t}USDT"

# -------------------- sizing --------------------

def position_size(entry: float, stop: float, equity: float, cash: float, strong: bool) -> float:
    risk_dollars = equity * 0.012  # 1.2% per trade
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = 0.60 if strong else 0.30
    usd = min(usd, equity * cap, cash)
    return max(0.0, float(usd))

# -------------------- main --------------------

def main() -> None:
    # Robust ENV parsing (avoid '' issue)
    try:
        equity = float(os.getenv("EQUITY") or "41000")
    except ValueError:
        equity = 41000.0
    try:
        cash = float(os.getenv("CASH") or "32000")
    except ValueError:
        cash = 32000.0

    snapshot = {
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": []
    }

    # Preservation / floor
    buffer_ok = (equity - FLOOR) >= BUFFER_MIN
    if (not buffer_ok) or equity <= 37_500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        write_signal("Raise cash now: exit weakest non-ETH, non-DOT positions on support breaks; halt new entries.", "A")
        write_snapshot(snapshot)
        print("Raise cash now: exit weakest non-ETH, non-DOT positions on support breaks; halt new entries.")
        return

    # (1) Regime
    regime = check_regime_btc_5m()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # (2) Universe (exclude ETH & DOT)
    mapping = load_revolut_mapping()
    universe: List[dict] = []
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if tkr in ("ETH", "DOT"):
            continue
        sym = symbol_on_binance(m)
        if not sym:
            continue
        md = binance_24h_and_book(sym)
        if not md["ok"]:
            continue
        spr = spread_pct(md["bid"], md["ask"])
        if (md["volume_usd"] >= 8_000_000) and (spr <= 0.005):
            universe.append({"ticker": tkr, "symbol": sym})

    snapshot["universe_count"] = len(universe)

    # (3 & 4) Scan signals
    scored: List[dict] = []
    for u in universe:
        bars = one_minute_bars(u["symbol"], limit=120)
        if not bars:
            continue
        br = aggressive_breakout(bars)
        if not br:
            continue
        mp = micro_pullback_ok(bars)
        if not mp:
            continue
        atr = atr_1m(bars, period=14)
        if atr <= 0:
            continue
        score = br["rvol"] * (1 + br["pct"])
        scored.append({
            "ticker": u["ticker"], "symbol": u["symbol"], "entry": mp["entry"],
            "pullback_low": mp["pullback_low"], "atr1m": atr,
            "rvol": br["rvol"], "pct": br["pct"], "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # No candidate → C
    if not scored:
        snapshot["candidates"] = []
        write_signal("Hold and wait.", "C")
        write_snapshot(snapshot)
        print("Hold and wait.")
        return

    # Best one → B
    top = scored[0]
    entry = float(top["entry"])
    stop = float(top["pullback_low"])
    atr = float(top["atr1m"])
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr

    buy_usd = position_size(entry, stop, equity, cash, strong_regime)
    snapshot["candidates"] = [top]
    write_snapshot(snapshot)

    sell_plan = "Sell weakest non-ETH, non-DOT on support break if cash needed."
    lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min",
        f"• What to sell from portfolio (excluding ETH, DOT): {sell_plan}",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}",
    ]
    msg = "\n".join(lines)
    write_signal(msg, "B")
    print(msg)

if __name__ == "__main__":
    main()