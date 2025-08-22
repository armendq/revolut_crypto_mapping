import os
import json
import time
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------- paths ----------
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)
SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNALS_PATH = Path("data/signals.json")

# ---------- helpers ----------
def _j(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

# ---------- exchange ----------
def binance_24h_and_book(symbol_usdt: str):
    try:
        t = _j(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol_usdt}")
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last
        ob = _j(f"https://api.binance.com/api/v3/depth?symbol={symbol_usdt}&limit=5")
        bid = float(ob["bids"][0][0]); ask = float(ob["asks"][0][0])
        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
    except Exception:
        return {"ok": False}

def binance_klines_1m(symbol_usdt: str, limit: int = 120):
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api3.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
    ]
    for u in urls:
        try:
            arr = _j(u)
            return [
                {'ts': x[0], 'o': float(x[1]), 'h': float(x[2]),
                 'l': float(x[3]), 'c': float(x[4]), 'v': float(x[5])}
                for x in arr
            ]
        except Exception:
            continue
    return []

# ---------- TA ----------
def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def true_range(h, l, pc):
    return max(h-l, abs(h-pc), abs(l-pc))

def atr_from_klines(bars: List[Dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period+1):
        h = bars[-i]['h']; l = bars[-i]['l']; pc = bars[-i-1]['c']
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

# ---------- regime check ----------
def check_regime() -> dict:
    bars = get_btc_5m_klines()
    if bars is None:
        return {"ok": False, "reason": "no-binance-klines"}
    if not isinstance(bars, pd.DataFrame):
        try:
            bars = pd.DataFrame(bars)
        except Exception:
            return {"ok": False, "reason": "no-binance-klines"}
    if bars.empty:
        return {"ok": False, "reason": "no-binance-klines"}

    if "close" not in bars.columns and "c" in bars.columns:
        bars = bars.rename(columns={"o": "open", "h": "high", "l": "low",
                                    "c": "close", "v": "volume"})

    close = bars["close"].astype(float)

    # VWAP fix
    vw = vwap(bars)
    if isinstance(vw, pd.Series):
        vwap_val = float(vw.iloc[-1])
    elif isinstance(vw, (list, tuple)) and len(vw) > 0:
        vwap_val = float(vw[-1])
    elif isinstance(vw, (int, float)):
        vwap_val = float(vw)
    elif "vwap" in bars.columns:
        vwap_val = float(bars["vwap"].astype(float).iloc[-1])
    else:
        vol = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series([1.0]*len(bars))
        vwap_series = (close * vol).cumsum() / vol.cumsum().replace(0, pd.NA)
        vwap_val = float(vwap_series.iloc[-1])

    ema9 = float(ema(close, span=9).iloc[-1])
    last_close = float(close.iloc[-1])

    ok = (last_close > vwap_val) and (last_close > ema9)

    return {"ok": ok,
            "reason": "" if ok else "btc-below-vwap-ema",
            "last_close": last_close,
            "vwap": vwap_val,
            "ema9": ema9}

# ---------- helpers for signals ----------
def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

def write_signal(sig_type: str, text: str):
    SIGNALS_PATH.write_text(json.dumps({"type": sig_type, "text": text}, indent=2))
    print(text)

# ---------- main ----------
def main():
    equity = float(os.getenv("EQUITY", "41000"))
    cash   = float(os.getenv("CASH", "32000"))
    floor  = 36500.0
    buffer_ok = (equity - floor) >= 1000.0

    snapshot = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": []
    }

    # Floor preservation
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "equity<=37500" if equity <= 37500.0 else "buffer<1000_over_floor"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        write_signal("A", "Raise cash now: exit weakest non-ETH, non-DOT; halt new entries.")
        return

    # Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # Universe scan (placeholder - you’ll extend with mapping + Binance checks)
    snapshot["universe_count"] = 0
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
    write_signal("C", "Hold and wait.")

if __name__ == "__main__":
    main()