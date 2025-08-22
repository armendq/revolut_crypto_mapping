# scripts/analyses.py
import os
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
import statistics
import pandas as pd

# local helpers
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------------- Paths ----------------
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNALS_PATH = Path("data/signals.json")
MAPPING_PATH = Path("data/revolut_mapping.json")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_snapshot(payload: Dict[str, Any]) -> None:
    SNAP_PATH.write_text(json.dumps(payload, indent=2))

def _write_signal(type_: str, text: str) -> None:
    SIGNALS_PATH.write_text(json.dumps(
        {"time": _now_iso(), "type": type_, "text": text}, indent=2
    ))

# ---------------- HTTP helper ----------------
def _j(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

# ---------------- Binance adapters ----------------
def binance_24h_and_book(symbol_usdt: str) -> Dict[str, Any]:
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

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict[str, float]]:
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
                {"ts": x[0], "o": float(x[1]), "h": float(x[2]),
                 "l": float(x[3]), "c": float(x[4]), "v": float(x[5])}
                for x in arr
            ]
        except Exception:
            continue
    return []

# ---------------- TA utils ----------------
def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict[str, float]], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]; l = bars[-i]["l"]; pc = bars[-i-1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

# ---------------- Rules ----------------
def check_regime() -> dict:
    bars = get_btc_5m_klines()

    # explicit empty check + normalize to DataFrame
    if bars is None:
        return {"ok": False, "reason": "no-binance-klines"}
    if not isinstance(bars, pd.DataFrame):
        try:
            bars = pd.DataFrame(bars)
        except Exception:
            return {"ok": False, "reason": "no-binance-klines"}

    if bars.empty:
        return {"ok": False, "reason": "no-binance-klines"}

    # normalize column names if they come as crypto-style (o,h,l,c,v)
    if "close" not in bars.columns and "c" in bars.columns:
        bars = bars.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})

    close = bars["close"].astype(float)

    # ---- VWAP: accept many shapes, always reduce to a single float (last) ----
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
        # fallback: compute rolling cumulative VWAP and take last
        vol = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series([1.0]*len(bars))
        vwap_series = (close * vol).cumsum() / vol.cumsum().replace(0, pd.NA)
        vwap_val = float(vwap_series.iloc[-1])

    # 9-EMA (last value as float)
    ema9 = float(ema(close, span=9).iloc[-1])

    last_close = float(close.iloc[-1])
    ok = (last_close > vwap_val) and (last_close > ema9)

    return {
        "ok": ok,
        "reason": "" if ok else "btc-below-vwap-ema",
        "last_close": last_close,
        "vwap": vwap_val,
        "ema9": ema9,
    }

def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
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
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last_c": last["c"]}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "pullback_low": last["l"]}
    return None

# ---------------- Universe ----------------
def _normalize_mapping_item(x: Any) -> Dict[str, str]:
    if isinstance(x, dict):
        return {"ticker": str(x.get("ticker") or "").upper(),
                "binance": str(x.get("binance") or "").upper()}
    if isinstance(x, str):
        return {"ticker": x.upper(), "binance": ""}
    return {"ticker": "", "binance": ""}

def load_revolut_mapping() -> List[Dict[str, str]]:
    if not MAPPING_PATH.exists():
        return []
    try:
        raw = json.loads(MAPPING_PATH.read_text())
        if isinstance(raw, list):
            return [_normalize_mapping_item(i) for i in raw if i]
    except Exception:
        return []
    return []

def best_binance_symbol(entry: Dict[str, str]) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = entry.get("ticker", "").upper()
    return f"{t}USDT" if t else None

# ---------------- Position sizing ----------------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * 0.012
    dist = max(entry - stop, 1e-8)
    qty = risk_dollars / dist
    usd = qty * entry

    alloc_cap = 0.60 if strong_regime else 0.30
    max_alloc = equity * alloc_cap
    usd = min(usd, max_alloc, cash)
    return max(0.0, usd)

# ---------------- Main ----------------
def main():
    equity = float(os.getenv("EQUITY", "41000"))
    cash   = float(os.getenv("CASH", "32000"))
    floor  = 36500.0
    buffer_ok = (equity - floor) >= 1000.0

    snapshot = {"time": _now_iso(), "equity": equity, "cash": cash,
                "breach": False, "breach_reason": "",
                "regime": {}, "universe_count": 0, "candidates": []}

    # Preservation
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        _write_snapshot(snapshot)
        _write_signal("A", "Raise cash now: exit weakest non-ETH, non-DOT positions; halt new entries.")
        return

    # Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # Universe
    mapping = load_revolut_mapping()
    universe = []
    for m in mapping:
        tkr = m.get("ticker", "")
        if not tkr or tkr in ("ETH", "DOT"):
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
    candidates = []
    for u in universe:
        bars = binance_klines_1m(u["symbol"])
        if not bars:
            continue
        br = aggressive_breakout(bars)
        mp = micro_pullback_ok(bars)
        if not br or not mp:
            continue
        atr1m = atr_from_klines(bars)
        if atr1m <= 0:
            continue
        score = br["rvol"] * (1 + br["pct"])
        candidates.append({**u, "entry": mp["entry"],
                           "pullback_low": mp["pullback_low"],
                           "atr1m": atr1m, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    snapshot["candidates"] = candidates[:1]
    _write_snapshot(snapshot)

    if not candidates:
        _write_signal("C", "Hold and wait.")
        return

    # Best candidate
    top = candidates[0]
    entry, stop, atr = float(top["entry"]), float(top["pullback_low"]), float(top["atr1m"])
    t1, t2 = entry + 0.8 * atr, entry + 1.5 * atr
    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    sell_plan = "Sell weakest non-ETH, non-DOT if cash needed."
    lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}",
        f"• Stop-loss: below {stop:.6f}",
        f"• What to sell: {sell_plan}",
        f"• Exact USD buy amount: ${buy_usd:,.2f}",
    ]
    _write_signal("B", "\n".join(lines))

if __name__ == "__main__":
    main()