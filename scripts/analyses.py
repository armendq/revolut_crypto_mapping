# scripts/analyses.py
# Stable scan + signals with ATR, targets, sizing, and debug logs.

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import urllib.request

# --------- Paths / constants ---------
DATA_DIR = Path("data")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)

SNAP_PATH   = ARTIFACTS / "market_snapshot.json"
DEBUG_PATH  = ARTIFACTS / "debug_scan.json"
SIGNALS_PATH = DATA_DIR / "signals.json"

EQUITY = float(os.getenv("EQUITY", "41000"))
CASH   = float(os.getenv("CASH", "32000"))

# --------- Small helpers ---------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _get_json(url: str, timeout: int = 20) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

def _get_json_list(url: str, timeout: int = 20) -> Optional[list]:
    req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None

# --------- Market data ---------
def binance_24h_and_book(symbol_usdt: str) -> Dict:
    """
    Return dict:
      {'ok': True, 'price': float, 'volume_usd': float, 'bid': float, 'ask': float}
       or {'ok': False, 'reason': ...}
    """
    t = _get_json(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol_usdt}")
    if not t:
        return {"ok": False, "reason": "no-24h"}

    try:
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last
    except Exception:
        return {"ok": False, "reason": "bad-24h"}

    # NOTE: fixed (no stray brace at end)
    ob = _get_json(f"https://api.binance.com/api/v3/depth?symbol={symbol_usdt}&limit=5")
    if not ob or not ob.get("bids") or not ob.get("asks"):
        return {"ok": False, "reason": "no-book"}

    try:
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])
    except Exception:
        return {"ok": False, "reason": "bad-book"}

    return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict]:
    """Return list of bars [{'ts':ms,'o','h','l','c','v'}] (best-effort)."""
    for host in ("api", "api1", "api2", "api3"):
        arr = _get_json_list(
            f"https://{host}.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}"
        )
        if isinstance(arr, list):
            out = []
            try:
                for x in arr:
                    out.append(
                        {"ts": x[0], "o": float(x[1]), "h": float(x[2]),
                         "l": float(x[3]), "c": float(x[4]), "v": float(x[5])}
                    )
                return out
            except Exception:
                continue
    return []

def get_btc_5m_klines(limit: int = 200) -> pd.DataFrame:
    """BTCUSDT 5m klines → DataFrame(time, open, high, low, close, volume)."""
    arr = _get_json_list(
        f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit={limit}"
    )
    if not isinstance(arr, list) or not arr:
        return pd.DataFrame()

    cols = ["open_time","o","h","l","c","v","close_time","q","n","taker_b","taker_q","i"]
    try:
        df = pd.DataFrame(arr, columns=cols)
        df = df.assign(
            time   = pd.to_datetime(df["open_time"], unit="ms", utc=True),
            open   = df["o"].astype(float),
            high   = df["h"].astype(float),
            low    = df["l"].astype(float),
            close  = df["c"].astype(float),
            volume = df["v"].astype(float),
        )[["time","open","high","low","close","volume"]]
        return df
    except Exception:
        return pd.DataFrame()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def vwap(df: pd.DataFrame) -> float:
    if df.empty: return 0.0
    pv = ((df["high"] + df["low"] + df["close"]) / 3.0) * df["volume"]
    vol = df["volume"].sum()
    return float(pv.sum() / vol) if vol > 0 else 0.0

# --------- TA helpers ---------
def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]
        l = bars[-i]["l"]
        pc = bars[-i - 1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

# --------- Signal rules ---------
def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid

def aggressive_breakout(bars_1m: List[Dict]) -> Optional[Dict]:
    """
    1m close +1.8%..+4% vs prev, RVOL >= 4 vs median of last 15 bars, and closes above 15m high.
    """
    if len(bars_1m) < 20:
        return None
    last, prev = bars_1m[-1], bars_1m[-2]
    try:
        pct = (last["c"] / prev["c"]) - 1.0
    except Exception:
        return None
    if not (0.018 <= pct <= 0.04):
        return None
    prior = bars_1m[-16:-1]
    if not prior:
        return None
    med = sorted([b["v"] for b in prior])[len(prior)//2]
    rvol = (last["v"] / med) if med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last": last}

def micro_pullback_ok(bars_1m: List[Dict]) -> Optional[Dict]:
    """
    Approximate 15–45s tight pullback by requiring small fade and micro range on last bar.
    """
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] <= 0 or last["c"] <= 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "pullback_low": last["l"]}
    return None

# --------- Mapping / universe ---------
def load_revolut_mapping() -> List[Dict]:
    with open(DATA_DIR / "revolut_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)

def best_binance_symbol(entry: Dict) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# --------- Position sizing ---------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    """
    Risk 1.2% of equity per trade; cap allocation at 60% (strong) or 30% (weak); never exceed cash.
    """
    risk_dollars = equity * 0.012
    dist = max(entry - stop, 1e-8)
    qty = risk_dollars / dist
    usd = qty * entry
    alloc_cap = equity * (0.60 if strong_regime else 0.30)
    return max(0.0, min(usd, alloc_cap, cash))

# --------- Debug collector ---------
DEBUG_LOG: List[Dict] = []
def _log(ticker: str, stage: str, reason: str, extra: Optional[Dict] = None):
    row = {"ticker": ticker, "stage": stage, "reason": reason}
    if extra:
        row.update(extra)
    DEBUG_LOG.append(row)

# --------- Main ---------
def main():
    # Regime via BTC 5m vs EMA9 & VWAP
    btc = get_btc_5m_klines()
    if btc.empty:
        regime = {"ok": False, "reason": "no-btc-5m"}
    else:
        last = float(btc["close"].iloc[-1])
        v = float(vwap(btc))
        e9 = float(ema(btc["close"], span=9).iloc[-1])
        ok = (last > v) and (last > e9)
        regime = {"ok": bool(ok), "reason": "btc-below-vwap-or-ema" if not ok else "ok",
                  "last": last, "vwap": v, "ema9": e9}

    snapshot = {
        "time": _now_iso(),
        "equity": EQUITY,
        "cash": CASH,
        "breach": False,
        "breach_reason": "",
        "regime": regime,
        "universe_count": 0,
        "candidates": []
    }

    # Capital preservation guard
    if EQUITY <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNALS_PATH.write_text(json.dumps({"type": "A", "text": "Raise cash now."}, indent=2))
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        print("Raise cash now.")
        return

    # Build universe (exclude ETH & DOT from rotation)
    universe: List[Dict] = []
    for m in load_revolut_mapping():
        if isinstance(m, str):
            # guard against unexpected lines
            continue
        tkr = (m.get("ticker") or "").upper()
        if not tkr:
            continue
        if tkr in ("ETH", "DOT"):
            _log(tkr, "universe", "excluded-eth-or-dot")
            continue

        symbol = best_binance_symbol(m)
        if not symbol:
            _log(tkr, "universe", "no-symbol")
            continue

        md = binance_24h_and_book(symbol)
        if not md.get("ok"):
            _log(tkr, "liquidity", md.get("reason", "no-24h/book"))
            continue

        spr = spread_pct(md["bid"], md["ask"])
        if md["volume_usd"] < 8_000_000:
            _log(tkr, "liquidity", "low-volume", {"vol_usd": md["volume_usd"]})
            continue
        if spr > 0.005:
            _log(tkr, "liquidity", "wide-spread", {"spread": spr})
            continue

        universe.append({
            "ticker": tkr,
            "symbol": symbol,
            "price": md["price"],
            "bid": md["bid"],
            "ask": md["ask"],
            "spread": spr,
            "vol_usd": md["volume_usd"],
        })

    snapshot["universe_count"] = len(universe)

    # Scan signals
    cands: List[Dict] = []
    for u in universe:
        bars = binance_klines_1m(u["symbol"], limit=120)
        if not bars:
            _log(u["ticker"], "signals", "no-klines")
            continue

        br = aggressive_breakout(bars)
        if not br:
            _log(u["ticker"], "signals", "no-breakout")
            continue

        mp = micro_pullback_ok(bars)
        if not mp:
            _log(u["ticker"], "signals", "no-micro-pullback")
            continue

        atr1m = atr_from_klines(bars, period=14)
        if atr1m <= 0:
            _log(u["ticker"], "signals", "no-atr")
            continue

        cands.append({
            "ticker": u["ticker"],
            "symbol": u["symbol"],
            "entry": mp["entry"],
            "stop": mp["pullback_low"],
            "atr1m": atr1m,
            "rvol": br["rvol"],
            "pct": br["pct"],
            "score": br["rvol"] * (1.0 + br["pct"]),
        })
        _log(u["ticker"], "signals", "candidate",
             {"entry": mp["entry"], "stop": mp["pullback_low"], "rvol": br["rvol"], "pct": br["pct"]})

    # Sort by score (desc)
    cands.sort(key=lambda x: x["score"], reverse=True)
    snapshot["candidates"] = cands
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
    DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))

    # Decide final signal
    if not cands:
        SIGNALS_PATH.write_text(json.dumps({"type": "C", "text": "Hold and wait."}, indent=2))
        print("Hold and wait. (No qualified candidates.)")
        return

    strong_regime = bool(regime.get("ok", False))
    top = cands[0]
    entry = top["entry"]
    stop  = top["stop"]
    atr   = top["atr1m"]

    # Targets (simple)
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr

    buy_usd = position_size(entry, stop, EQUITY, CASH, strong_regime)

    txt = (
        f"• {top['ticker']} + {top['ticker']}\n"
        f"• Entry price: {entry:.6f}\n"
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R\n"
        f"• Stop-loss / exit plan: Below {stop:.6f}\n"
        f"• What to sell from portfolio (excluding ETH, DOT): Sell weakest non-ETH/DOT on support break if cash needed.\n"
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    )

    if strong_regime and buy_usd > 0:
        SIGNALS_PATH.write_text(json.dumps({"type": "B", "text": txt}, indent=2))
        print("Signal B generated:\n" + txt)
    else:
        SIGNALS_PATH.write_text(json.dumps({"type": "C", "text": "Candidates found but regime/cash not aligned. Hold."}, indent=2))
        print("Candidates found but regime/cash not aligned. Hold.")

if __name__ == "__main__":
    main()