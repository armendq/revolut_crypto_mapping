# scripts/analyses.py
# End-to-end scan that produces one of A/B/C and writes data/market_snapshot.json + data/signals.json

import os
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import urllib.request
import pandas as pd

# --------------------------- Paths ---------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
SNAP_PATH = DATA_DIR / "market_snapshot.json"
SIGNALS_PATH = DATA_DIR / "signals.json"
MAP_PATH = DATA_DIR / "revolut_mapping.json"

# --------------------------- HTTP helpers ---------------------------
def _j(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --------------------------- TA helpers ---------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def vwap_df(df: pd.DataFrame) -> float:
    # expects columns: high, low, close, volume
    if df.empty:
        return 0.0
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).sum()
    vv = df["volume"].sum()
    return float(pv / vv) if vv > 0 else 0.0

def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]; l = bars[-i]["l"]; pc = bars[-i-1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs) if trs else 0.0

# --------------------------- Binance lightweight ---------------------------
def binance_klines(symbol: str, interval: str, limit: int = 150) -> List[List]:
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api3.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    ]
    for u in urls:
        try:
            return _j(u)
        except Exception:
            continue
    return []

def binance_1m_struct(symbol: str, limit: int = 120) -> List[Dict]:
    arr = binance_klines(symbol, "1m", limit)
    out = []
    for x in arr:
        out.append({
            "ts": x[0],
            "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
            "c": float(x[4]), "v": float(x[5])
        })
    return out

def binance_5m_df(symbol: str, limit: int = 100) -> pd.DataFrame:
    arr = binance_klines(symbol, "5m", limit)
    if not arr:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(arr, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["open","high","low","close","volume"]]

def binance_24h_and_book(symbol: str) -> Dict:
    try:
        t = _j(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last
        ob = _j(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=5")
        bid = float(ob["bids"][0][0]); ask = float(ob["asks"][0][0])
        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
    except Exception:
        return {"ok": False}

# --------------------------- Strategy rules ---------------------------
def check_regime() -> Dict:
    # BTCUSDT 5m close > VWAP and > 9-EMA
    df = binance_5m_df("BTCUSDT", limit=120)
    if df.empty:
        return {"ok": False, "reason": "no-btc-5m"}
    close = df["close"]
    vwap_val = vwap_df(df)
    ema9 = float(ema(close, span=9).iloc[-1])
    last = float(close.iloc[-1])
    ok = (last > vwap_val) and (last > ema9)
    return {
        "ok": bool(ok),
        "reason": "" if ok else "btc-below-vwap-or-ema",
        "last": last, "vwap": vwap_val, "ema9": ema9
    }

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid

def median(vals: List[float]) -> float:
    return statistics.median(vals) if vals else 0.0

def aggressive_breakout(bars_1m: List[Dict]) -> Optional[Dict]:
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]; prev = bars_1m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    if not (0.018 <= pct <= 0.04):
        return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]])
    rvol = (last["v"] / vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last": last}

def micro_pullback_ok(bars_1m: List[Dict]) -> Optional[Dict]:
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] == 0 or last["c"] == 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "pullback_low": last["l"]}
    return None

def atr_1m(bars_1m: List[Dict], period: int = 14) -> float:
    return atr_from_klines(bars_1m, period=period)

# --------------------------- Mapping helpers ---------------------------
def load_revolut_mapping() -> List:
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def normalize_mapping(raw: List) -> List[Dict]:
    """
    Accepts a list of dicts OR strings and returns
    [{'ticker': 'ARB', 'binance': 'ARBUSDT'|None}, ...] unique by ticker.
    """
    norm = []
    import re
    for x in raw:
        if isinstance(x, dict):
            t = (x.get("ticker") or x.get("symbol") or x.get("code") or "").strip().upper()
            b = (x.get("binance") or "").strip().upper() or None
        else:
            s = str(x).strip().upper()
            tokens = re.split(r"[^A-Z0-9]+", s)
            cand = [tok for tok in tokens if tok]
            t = cand[-1] if cand else ""
            b = None
        if t:
            norm.append({"ticker": t, "binance": b})
    # de-dup
    out, seen = [], set()
    for e in norm:
        if e["ticker"] not in seen:
            out.append(e); seen.add(e["ticker"])
    return out

def best_binance_symbol(entry: Dict) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = entry.get("ticker")
    return f"{t}USDT" if t else None

# --------------------------- Position sizing ---------------------------
def position_size(entry: float, stop: float, equity: float, cash: float, strong: bool) -> float:
    risk_dollars = equity * 0.012
    dist = max(entry - stop, 1e-8)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = 0.60 if strong else 0.30
    usd = min(usd, equity * cap, cash)
    return max(0.0, usd)

# --------------------------- Outputs ---------------------------
def write_signal_A():
    SIGNALS_PATH.write_text(json.dumps({
        "type": "A",
        "text": "Raise cash now: exit weakest non-ETH, non-DOT on support breaks; halt new entries."
    }, indent=2))

def write_signal_B(ticker: str, entry: float, t1: float, t2: float, stop: float, buy_usd: float):
    SIGNALS_PATH.write_text(json.dumps({
        "type": "B",
        "coin": ticker,
        "entry": round(entry, 6),
        "target": {"T1": round(t1, 6), "T2": round(t2, 6), "trail_after_R": 1.0},
        "stop": round(stop, 6),
        "sell_plan": "Sell weakest non-ETH, non-DOT on support break if cash needed.",
        "buy_usd": round(buy_usd, 2)
    }, indent=2))

def write_signal_C():
    SIGNALS_PATH.write_text(json.dumps({"type": "C", "text": "Hold and wait."}, indent=2))

# --------------------------- MAIN ---------------------------
def main():
    equity = float(os.getenv("EQUITY", "41000") or 0)
    cash   = float(os.getenv("CASH", "32000") or 0)
    floor = 36500.0
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

    # Preservation / buffer checks
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        write_signal_A()
        print("A) Preservation breach → rotating/raising cash.")
        return

    # Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # Universe
    raw_map = load_revolut_mapping()
    mapping = normalize_mapping(raw_map)

    universe = []
    for m in mapping:
        tkr = m["ticker"]
        if tkr in ("ETH", "DOT"):
            continue
        sym = best_binance_symbol(m)
        if not sym:
            continue
        md = binance_24h_and_book(sym)
        if not md.get("ok"):
            continue
        spr = spread_pct(md["bid"], md["ask"])
        if md["volume_usd"] >= 8_000_000 and spr <= 0.005:
            universe.append({"ticker": tkr, "symbol": sym})

    snapshot["universe_count"] = len(universe)

    # Signals
    best = None
    for u in universe:
        bars = binance_1m_struct(u["symbol"], limit=120)
        if not bars:
            continue
        br = aggressive_breakout(bars)
        if not br:
            continue
        mp = micro_pullback_ok(bars)
        if not mp:
            continue
        atr1m = atr_1m(bars, period=14)
        if atr1m <= 0:
            continue
        score = br["rvol"] * (1 + br["pct"])
        cand = {
            "ticker": u["ticker"],
            "symbol": u["symbol"],
            "entry": mp["entry"],
            "stop": mp["pullback_low"],
            "atr1m": atr1m,
            "score": score
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand

    # Write snapshot (minimal)
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))

    if best is None:
        write_signal_C()
        print('C) "Hold and wait."')
        return

    # Only top 1 candidate is acted on
    entry = best["entry"]; stop = best["stop"]; atr = best["atr1m"]
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr
    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    write_signal_B(best["ticker"], entry, t1, t2, stop, buy_usd)

    # Print in the exact B format (single candidate)
    print("\n".join([
        f"• {best['ticker']} + {best['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min",
        "• What to sell from portfolio (excluding ETH, DOT): Sell weakest non-ETH, non-DOT on support break if cash needed.",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    ]))

if __name__ == "__main__":
    main()