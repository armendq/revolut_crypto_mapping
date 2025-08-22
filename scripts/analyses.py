# scripts/analyses.py

# --- put near your imports ---
import os

def _fenv(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(v) if v and str(v).strip() else float(default)
    except Exception:
        return float(default)

# --- inside main() ---
equity = _fenv("EQUITY", 41000.0)
cash   = _fenv("CASH",   32000.0)

import json
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ------------ Paths -------------
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNALS_PATH = Path("data/signals.json")
MAPPING_JSON = Path("data/revolut_mapping.json")

# ------------ Small utils -------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _j(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def write_signal(sig_type: str, text: str, extra: Optional[Dict] = None):
    payload = {"type": sig_type, "text": text}
    if extra:
        payload.update(extra)
    SIGNALS_PATH.write_text(json.dumps(payload, indent=2))
    print(text)

# ------------ Lightweight exchange adapters -------------
def binance_24h_and_book(symbol_usdt: str) -> Dict:
    """Return {'ok', 'price', 'volume_usd', 'bid', 'ask'} or {'ok': False}."""
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

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict]:
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

# ------------ TA helpers -------------
def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def true_range(h, l, pc):
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]; l = bars[-i]["l"]; pc = bars[-i - 1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

# ------------ Strategy rule checks -------------
def check_regime() -> Dict:
    """BTC 5m close > VWAP AND > 9-EMA."""
    bars = get_btc_5m_klines()
    if bars is None:
        return {"ok": False, "reason": "no-binance-klines"}

    if isinstance(bars, list):
        bars = pd.DataFrame(bars)

    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return {"ok": False, "reason": "no-binance-klines"}

    # unify columns if needed
    cols = {c.lower(): c for c in bars.columns}
    if "close" not in cols and "c" in bars.columns:
        bars = bars.rename(columns={"o": "open", "h": "high", "l": "low",
                                    "c": "close", "v": "volume"})
    close = bars["close"].astype(float)

    # VWAP may return a series; always take the latest float
    vw = vwap(bars)
    if isinstance(vw, pd.Series):
        vwap_val = float(vw.iloc[-1])
    elif isinstance(vw, (list, tuple)) and vw:
        vwap_val = float(vw[-1])
    elif isinstance(vw, (int, float)):
        vwap_val = float(vw)
    elif "vwap" in bars.columns:
        vwap_val = float(bars["vwap"].astype(float).iloc[-1])
    else:
        vol = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series([1.0]*len(bars))
        vwap_val = float(((close * vol).cumsum() / vol.cumsum()).iloc[-1])

    ema9 = float(ema(close, span=9).iloc[-1])
    last_close = float(close.iloc[-1])
    ok = (last_close > vwap_val) and (last_close > ema9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-ema",
            "last_close": last_close, "vwap": vwap_val, "ema9": ema9}

def aggressive_breakout(bars_1m: List[Dict]) -> Optional[Dict]:
    """+1.8%..+4.0%, RVOL >= 4.0 vs 15m median, break above last 15m high."""
    if len(bars_1m) < 20:
        return None
    last, prev = bars_1m[-1], bars_1m[-2]
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
    """Approx micro pullback (<=0.6% from high, hold)."""
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

# ------------ Universe + mapping -------------
def load_revolut_mapping() -> List[Dict]:
    """Expect a list of dicts with keys like {'ticker','binance','name'}."""
    try:
        raw = json.loads(MAPPING_JSON.read_text())
    except Exception:
        return []
    out = []
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, str):
                out.append({"ticker": x})
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                out.append(v)
            else:
                out.append({"ticker": k, "binance": v if isinstance(v, str) else None})
    return out

def best_binance_symbol(entry: Dict) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    if not t:
        return None
    return f"{t}USDT"

# ------------ Position sizing -------------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * 0.012  # 1.2%
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    alloc_cap = 0.60 if strong_regime else 0.30
    usd = min(usd, equity * alloc_cap, cash)
    return max(0.0, usd)

# ------------ Main -------------
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

    # Preservation checks (Rule 5 & 7)
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "equity<=37500" if equity <= 37500.0 else "buffer<1000_over_floor"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        write_signal("A", "Raise cash now: exit weakest non-ETH, non-DOT on support breaks; halt new entries.")
        return

    # Regime (Rule 1)
    regime = check_regime()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # Universe (Rule 2)
    mapping = load_revolut_mapping()
    universe = []
    for m in mapping:
        if not isinstance(m, dict):
            continue
        tkr = (m.get("ticker") or "").upper()
        if tkr in ("ETH", "DOT") or not tkr:
            continue
        sym = best_binance_symbol(m)
        if not sym:
            continue
        md = binance_24h_and_book(sym)
        if not md.get("ok"):
            continue
        spr = spread_pct(md["bid"], md["ask"])
        if md["volume_usd"] >= 8_000_000 and spr <= 0.005:
            universe.append({
                "ticker": tkr,
                "binance": sym,
                "price": md["price"],
                "bid": md["bid"],
                "ask": md["ask"],
                "spread": spr,
                "vol_usd": md["volume_usd"]
            })

    snapshot["universe_count"] = len(universe)

    # Signals (Rules 3 & 4)
    scored: List[Dict] = []
    for u in universe:
        bars = binance_klines_1m(u["binance"], limit=120)
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
        score = br["rvol"] * (1 + br["pct"])  # simple priority
        scored.append({
            "ticker": u["ticker"],
            "symbol": u["binance"],
            "entry": mp["entry"],
            "pullback_low": mp["pullback_low"],
            "atr1m": atr1m,
            "rvol": br["rvol"],
            "pct": br["pct"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # No candidate
    if not scored:
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        write_signal("C", "Hold and wait.")
        return

    # Best single candidate (Rule 1 weak regime still takes top1; sizing caps allocation)
    top = scored[0]
    entry = float(top["entry"])
    stop  = float(top["pullback_low"])
    atr   = float(top["atr1m"])

    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr
    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    snapshot["candidates"] = [top]
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))

    # Compose Option B text exactly as requested
    sell_plan = "Sell weakest non-ETH, non-DOT on support break if cash needed."
    b_text = (
        f"• {top['ticker']} + {top['ticker']}\n"
        f"• Entry price: {entry:.6f}\n"
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R\n"
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min\n"
        f"• What to sell from portfolio (excluding ETH, DOT): {sell_plan}\n"
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    )
    write_signal("B", b_text)

if __name__ == "__main__":
    main()