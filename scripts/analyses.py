# scripts/analyses.py
# Robust scanner for Revolut-X universe using public market data
# Emits exactly one of:
#  A) preservation breach → rotation/raise-cash instructions
#  B) single best A+ candidate with full plan
#  C) "Hold and wait."

import os
import json
import urllib.request
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd

# Local utilities: BTC 5m klines, EMA, VWAP
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------- constants & paths ----------
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)
SNAP_PATH = ARTIFACTS / "market_snapshot.json"

# ---------- tiny HTTP helper ----------
def _j(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15) -> Any:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------- light exchange adapters ----------
def binance_24h_and_book(symbol_usdt: str) -> Dict[str, Any]:
    """
    Returns:
      {'ok': bool, 'price': float, 'volume_usd': float, 'bid': float, 'ask': float}
    """
    try:
        t = _j(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol_usdt}")
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last

        ob = _j(f"https://api.binance.com/api/v3/depth?symbol={symbol_usdt}&limit=5")
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])

        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
    except Exception:
        return {"ok": False}

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict[str, float]]:
    """
    Returns list of dicts: [{'ts':ms,'o':...,'h':...,'l':...,'c':...,'v':...}]
    """
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
                {"ts": x[0], "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
                 "c": float(x[4]), "v": float(x[5])}
                for x in arr
            ]
        except Exception:
            continue
    return []

# ---------- TA utils ----------
def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict[str, float]], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]; l = bars[-i]["l"]; pc = bars[-i - 1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

# ---------- rule checks ----------
def check_regime() -> Dict[str, Any]:
    """
    Rule 1: BTC 5m close > 5m VWAP AND > 9-EMA.
    """
    bars: Optional[pd.DataFrame] = get_btc_5m_klines()
    if bars is None or bars.empty:
        return {"ok": False, "reason": "no-binance-klines"}

    close = bars["close"]
    last_close = float(close.iloc[-1])

    # Ensure scalar values
    vwap_series = vwap(bars)
    vwap_val = float(vwap_series.iloc[-1]) if hasattr(vwap_series, "iloc") else float(vwap_series)
    ema_val = float(ema(close, span=9).iloc[-1])

    ok = (last_close > vwap_val) and (last_close > ema_val)
    return {"ok": bool(ok), "last": last_close, "vwap": vwap_val, "ema9": ema_val}

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Rule 3:
      - 1-min price +1.8% to +4.0% vs previous close,
      - RVOL >= 4.0 vs last 15-min median,
      - Break above last 15-min high with follow-through.
    """
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

    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last_close": last["c"]}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Rule 4 approximation (no tick feed):
      - (high - close)/high <= 0.006
      - (high - low)/high <= 0.006
      -> entry = last close, stop = last low
    """
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

# ---------- mapping / universe ----------
def _coerce_entry(x: Any) -> Dict[str, Any]:
    """
    Accepts:
      - dict with fields (ticker, binance, ...) → passes through
      - string ticker → converts to {'ticker': 'XYZ'}
    """
    if isinstance(x, dict):
        # normalize keys
        out = {k.lower(): v for k, v in x.items()}
        if "ticker" in out:
            out["ticker"] = str(out["ticker"]).upper()
        return out
    if isinstance(x, str):
        return {"ticker": x.upper()}
    return {}

def load_revolut_mapping() -> List[Dict[str, Any]]:
    """
    Loads mapping from JSON or CSV (whichever exists).
    Supports:
      - JSON list of dicts
      - JSON list of tickers (strings)
      - CSV with a 'ticker' column
    """
    json_path = Path("data/revolut_mapping.json")
    csv_path = Path("data/revolut_mapping.csv")

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return [ _coerce_entry(x) for x in data ]
        except Exception:
            pass

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            cols = {c.lower(): c for c in df.columns}
            if "ticker" in cols:
                return [ {"ticker": str(t).upper()} for t in df[cols["ticker"]].tolist() ]
            # else: treat each row as dict
            recs: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                d = {str(k).lower(): row[k] for k in df.columns}
                if "ticker" in d:
                    d["ticker"] = str(d["ticker"]).upper()
                recs.append(d)
            return recs
        except Exception:
            pass

    # Nothing found → empty
    return []

def best_binance_symbol(entry: Dict[str, Any]) -> Optional[str]:
    """
    Prefer explicit mapping (entry['binance']), else <TICKER>USDT.
    """
    if entry.get("binance"):
        return str(entry["binance"]).upper()
    t = entry.get("ticker")
    if not t:
        return None
    return f"{t}USDT"

# ---------- sizing ----------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    """
    Risk 1.2% of equity. Allocation cap 60% (strong) / 30% (weak). Respect cash.
    """
    risk_dollars = equity * 0.012
    dist = max(entry - stop, 1e-8)
    qty = risk_dollars / dist
    usd = qty * entry

    alloc_cap = 0.60 if strong_regime else 0.30
    max_alloc = equity * alloc_cap
    usd = min(usd, max_alloc, cash)
    return max(0.0, usd)

# ---------- main ----------
def main() -> None:
    # Read portfolio envelope from env with safe defaults
    equity = float(os.getenv("EQUITY", "41000"))
    cash = float(os.getenv("CASH", "32000"))

    # Floor & buffer
    hard_floor = 36500.0
    buffer_ok = (equity - hard_floor) >= 1000.0

    snapshot = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": [],
    }

    # Preservation rules first
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        print("Raise cash now: exit weakest non-ETH, non-DOT positions on support breaks; halt new entries.")
        return

    # 1) Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # 2) Universe (volume/spread). Exclude ETH & DOT from rotation
    mapping = load_revolut_mapping()
    universe: List[Dict[str, Any]] = []

    for raw in mapping:
        m = _coerce_entry(raw)
        tkr = str(m.get("ticker", "")).upper()
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
            universe.append({
                "ticker": tkr,
                "symbol": sym,
                "price": md["price"],
                "bid": md["bid"],
                "ask": md["ask"],
                "spread": spr,
                "vol_usd": md["volume_usd"],
            })

    snapshot["universe_count"] = len(universe)

    # 3 & 4) Signals
    scored: List[Dict[str, Any]] = []
    for u in universe:
        bars = binance_klines_1m(u["symbol"], limit=120)
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
        scored.append({
            "ticker": u["ticker"],
            "symbol": u["symbol"],
            "entry": mp["entry"],
            "pullback_low": mp["pullback_low"],
            "atr1m": atr1m,
            "rvol": br["rvol"],
            "pct": br["pct"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # No setups → C
    if not scored:
        snapshot["candidates"] = []
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        print("Hold and wait.")
        return

    # Take only the top 1 always (fits weak-regime constraint automatically)
    top = scored[0]
    entry = float(top["entry"])
    stop = float(top["pullback_low"])
    atr = float(top["atr1m"])

    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr

    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    snapshot["candidates"] = [top]
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))

    # Output B in the exact bullet format
    sell_plan = "Sell weakest non-ETH, non-DOT on support break if cash needed."
    lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min",
        f"• What to sell from portfolio (excluding ETH, DOT): {sell_plan}",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}",
    ]
    print("\n".join(lines))

if __name__ == "__main__":
    main()