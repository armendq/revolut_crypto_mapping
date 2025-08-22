# scripts/analyses.py
# safe, verbose scan with debug logging

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import urllib.request

# ---------- PATHS / CONSTANTS ----------
ARTIFACTS = Path("artifacts")
DATA_DIR = Path("data")
ARTIFACTS.mkdir(exist_ok=True, parents=True)

SNAP_PATH = ARTIFACTS / "market_snapshot.json"
DEBUG_PATH = ARTIFACTS / "debug_scan.json"
SIGNALS_PATH = DATA_DIR / "signals.json"

EQUITY = float(os.getenv("EQUITY", "41000"))
CASH = float(os.getenv("CASH", "32000"))

# ---------- UTILS ----------
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

# ---------- MARKET DATA ----------
def binance_24h_and_book(symbol_usdt: str) -> Dict:
    """
    Return:
      {'ok': bool, 'price': float, 'volume_usd': float, 'bid': float, 'ask': float}
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

    # FIXED: removed stray '}' at the end of the URL
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
    """Return list of bars [{'ts':ms,'o':...,'h':...,'l':...,'c':...,'v':...}]"""
    hosts = ["api", "api1", "api2", "api3"]
    for h in hosts:
        url = f"https://{h}.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}"
        arr = _get_json_list(url)
        if isinstance(arr, list):
            out = []
            try:
                for x in arr:
                    out.append(
                        {"ts": x[0], "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
                         "c": float(x[4]), "v": float(x[5])}
                    )
                return out
            except Exception:
                continue
    return []

def get_btc_5m_klines(limit: int = 200) -> pd.DataFrame:
    """BTCUSDT 5m klines → DataFrame with columns time, open, high, low, close, volume"""
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit={limit}"
    arr = _get_json_list(url)
    if not isinstance(arr, list) or not arr:
        return pd.DataFrame()

    cols = ["open_time","o","h","l","c","v","close_time","q","n","taker_b","taker_q","i"]
    try:
        df = pd.DataFrame(arr, columns=cols)
        df = df.assign(
            time=pd.to_datetime(df["open_time"], unit="ms", utc=True),
            open=df["o"].astype(float),
            high=df["h"].astype(float),
            low=df["l"].astype(float),
            close=df["c"].astype(float),
            volume=df["v"].astype(float),
        )[["time","open","high","low","close","volume"]]
        return df
    except Exception:
        return pd.DataFrame()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def vwap(df: pd.DataFrame) -> float:
    """VWAP over the whole frame"""
    if df.empty:
        return 0.0
    pv = (df["high"] + df["low"] + df["close"]) / 3.0 * df["volume"]
    vol = df["volume"].sum()
    return float((pv.sum() / vol)) if vol > 0 else 0.0

# ---------- SIGNAL LOGIC ----------
def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid <= 0 else (ask - bid) / mid

def aggressive_breakout(bars_1m: List[Dict]) -> Optional[Dict]:
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
    med = sorted([b["v"] for b in prior])[len(prior)//2] if prior else 0.0
    rvol = (last["v"] / med) if med > 0 else 0.0
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
    if last["h"] <= 0 or last["c"] <= 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "pullback_low": last["l"]}
    return None

def load_revolut_mapping() -> List[Dict]:
    with open(DATA_DIR / "revolut_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)

def best_binance_symbol(entry: Dict) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# ---------- DEBUG COLLECTOR ----------
DEBUG_LOG: List[Dict] = []

def _log(ticker: str, stage: str, reason: str, extra: Optional[Dict] = None):
    item = {"ticker": ticker, "stage": stage, "reason": reason}
    if extra:
        item.update(extra)
    DEBUG_LOG.append(item)

# ---------- MAIN ----------
def main():
    # regime
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

    # floor guard
    if EQUITY <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNALS_PATH.write_text(json.dumps({"type": "A", "text": "Raise cash now."}, indent=2))
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        print("Raise cash now.")
        return

    # universe
    mapping = load_revolut_mapping()
    universe: List[Dict] = []
    for m in mapping:
        if isinstance(m, str):
            continue
        tkr = (m.get("ticker") or "").upper()
        if not tkr or tkr in ("ETH", "DOT"):
            if tkr in ("ETH", "DOT"):
                _log(tkr, "universe", "excluded-eth-or-dot")
            continue

        sym = best_binance_symbol(m)
        if not sym:
            _log(tkr, "universe", "no-symbol")
            continue

        md = binance_24h_and_book(sym)
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
            "binance": sym,
            "price": md["price"],
            "bid": md["bid"],
            "ask": md["ask"],
            "spread": spr,
            "vol_usd": md["volume_usd"]
        })

    snapshot["universe_count"] = len(universe)

    candidates: List[Dict] = []
    for u in universe:
        bars = binance_klines_1m(u["binance"], limit=120)
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

        candidates.append({
            "ticker": u["ticker"],
            "symbol": u["binance"],
            "entry": mp["entry"],
            "pullback_low": mp["pullback_low"],
            "rvol": br["rvol"],
            "pct": br["pct"]
        })
        _log(u["ticker"], "signals", "candidate",
             {"entry": mp["entry"], "stop": mp["pullback_low"], "rvol": br["rvol"], "pct": br["pct"]})

    snapshot["candidates"] = candidates

    # outputs
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
    DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))

    if not candidates:
        SIGNALS_PATH.write_text(json.dumps({"type": "C", "text": "Hold and wait."}, indent=2))
        print("Hold and wait. (No qualified candidates.)")
    else:
        if not regime.get("ok", False):
            SIGNALS_PATH.write_text(json.dumps(
                {"type": "C", "text": "Candidates found but regime weak. Hold."}, indent=2))
            print("Candidates found but regime is weak. Hold.")
        else:
            top = candidates[0]
            txt = (
                f"• {top['ticker']} + {top['ticker']}\n"
                f"• Entry price: {top['entry']:.6f}\n"
                f"• Target: trail after +1.0R\n"
                f"• Stop-loss / exit plan: Below {top['pullback_low']:.6f}"
            )
            SIGNALS_PATH.write_text(json.dumps({"type": "B", "text": txt}, indent=2))
            print("Signal B generated:\n" + txt)

if __name__ == "__main__":
    main()