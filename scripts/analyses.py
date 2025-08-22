# scripts/analyses.py
import os, json, urllib.request, statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------------- paths ----------------
ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True, parents=True)
SNAP_PATH = ARTIFACTS / "market_snapshot.json"
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True, parents=True)
SIGNALS_PATH = DATA_DIR / "signals.json"

# ---------------- tiny HTTP ----------------
def _j(url: str, headers: Optional[Dict[str,str]] = None, timeout: int = 15) -> Any:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------- exchange adapters ----------------
def binance_24h_and_book(symbol_usdt: str) -> Dict[str, Any]:
    try:
        t = _j(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol_usdt}")
        last = float(t["lastPrice"]); vol_base = float(t["volume"]); vol_usd = vol_base * last
        ob = _j(f"https://api.binance.com/api/v3/depth?symbol={symbol_usdt}&limit=5")
        bid = float(ob["bids"][0][0]); ask = float(ob["asks"][0][0])
        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask, "src":"binance"}
    except Exception:
        return {"ok": False}

def coinbase_24h_and_book(product_id: str) -> Dict[str, Any]:
    # product_id like "ARB-USD"
    try:
        stats = _j(f"https://api.exchange.coinbase.com/products/{product_id}/stats")
        ticker = _j(f"https://api.exchange.coinbase.com/products/{product_id}/ticker")
        book   = _j(f"https://api.exchange.coinbase.com/products/{product_id}/book?level=1")
        last = float(ticker["price"])
        vol_base = float(stats["volume"])
        vol_usd = vol_base * last
        bid = float(book["bids"][0][0]); ask = float(book["asks"][0][0])
        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask, "src":"coinbase"}
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
            return [{"ts": x[0], "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
                     "c": float(x[4]), "v": float(x[5])} for x in arr]
        except Exception:
            continue
    return []

# ---------------- TA utils ----------------
def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def true_range(h: float, l: float, pc: float) -> float:
    return max(h-l, abs(h-pc), abs(l-pc))

def atr_from_klines(bars: List[Dict[str, float]], period: int = 14) -> float:
    if len(bars) < period + 1: return 0.0
    trs = []
    for i in range(1, period+1):
        h = bars[-i]["h"]; l = bars[-i]["l"]; pc = bars[-i-1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs)/len(trs)

# ---------------- rules ----------------
def _btc_5m_df_fallback() -> Optional[pd.DataFrame]:
    """Fallback: CoinGecko OHLC (5m granularity for 1 day)."""
    try:
        arr = _j("https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=1")
        # arr: [timestamp, open, high, low, close]
        df = pd.DataFrame(arr, columns=["ts","open","high","low","close"])
        # keep last ~200 rows
        return df.tail(200)
    except Exception:
        return None

def check_regime() -> Dict[str, Any]:
    bars = get_btc_5m_klines()            # your helper (Binance)
    src = "binance"
    if bars is None or bars.empty:
        fb = _btc_5m_df_fallback()
        if fb is None or fb.empty:
            return {"ok": False, "reason": "no-btc-5m"}
        bars = fb; src = "coingecko"

    close = bars["close"]
    last_close = float(close.iloc[-1])
    vwap_series = vwap(bars)
    vwap_val = float(vwap_series.iloc[-1]) if hasattr(vwap_series, "iloc") else float(vwap_series)
    ema_val  = float(ema(close, span=9).iloc[-1])

    ok = (last_close > vwap_val) and (last_close > ema_val)
    return {"ok": bool(ok), "src": src, "last": last_close, "vwap": vwap_val, "ema9": ema_val}

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 20: return None
    last, prev = bars_1m[-1], bars_1m[-2]
    pct = (last["c"]/prev["c"]) - 1.0
    if not (0.018 <= pct <= 0.04): return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]])
    rvol = (last["v"]/vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0: return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15*1.0005: return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15, "last_close": last["c"]}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 2: return None
    last = bars_1m[-1]
    if last["h"] == 0 or last["c"] == 0: return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "pullback_low": last["l"]}
    return None

# ---------------- mapping & universe ----------------
def _coerce_entry(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        d = {k.lower(): v for k, v in x.items()}
        if "ticker" in d: d["ticker"] = str(d["ticker"]).upper()
        return d
    if isinstance(x, str):
        return {"ticker": x.upper()}
    return {}

def load_revolut_mapping() -> List[Dict[str, Any]]:
    j = Path("data/revolut_mapping.json"); c = Path("data/revolut_mapping.csv")
    if j.exists():
        try:
            return [_coerce_entry(x) for x in json.loads(j.read_text(encoding="utf-8"))]
        except Exception:
            pass
    if c.exists():
        try:
            df = pd.read_csv(c)
            if "ticker" in {col.lower() for col in df.columns}:
                col = [col for col in df.columns if col.lower()=="ticker"][0]
                return [{"ticker": str(t).upper()} for t in df[col].tolist()]
            recs = []
            for _, row in df.iterrows():
                d = {str(k).lower(): row[k] for k in df.columns}
                if "ticker" in d: d["ticker"] = str(d["ticker"]).upper()
                recs.append(d)
            return recs
        except Exception:
            pass
    return []

def best_binance_symbol(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("binance"): return str(entry["binance"]).upper()
    t = entry.get("ticker"); 
    return f"{t}USDT" if t else None

def best_coinbase_product(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("coinbase"): return str(entry["coinbase"]).upper()
    t = entry.get("ticker"); 
    return f"{t}-USD" if t else None

# ---------------- sizing ----------------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * 0.012
    dist = max(entry - stop, 1e-8)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = 0.60 if strong_regime else 0.30
    usd = min(usd, equity*cap, cash)
    return max(0.0, usd)

# ---------------- main ----------------
def _write_signal(type_: str, text: str) -> None:
    SIGNALS_PATH.write_text(json.dumps({"type": type_, "text": text}, indent=2))

def main() -> None:
    equity = float(os.getenv("EQUITY", "41000"))
    cash   = float(os.getenv("CASH", "32000"))
    floor  = 36500.0
    buffer_ok = (equity - floor) >= 1000.0

    snapshot = {
        "time": _now_iso(), "equity": equity, "cash": cash,
        "breach": False, "breach_reason": "", "regime": {},
        "universe_count": 0, "candidates": []
    }

    # Preservation first
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        text = "Raise cash now: exit weakest non-ETH, non-DOT positions on support breaks; halt new entries."
        print(text); _write_signal("A", text); return

    # 1) Regime
    regime = check_regime(); snapshot["regime"] = regime
    strong_regime = bool(regime.get("ok", False))

    # 2) Universe with Binance → Coinbase fallback
    mapping = load_revolut_mapping()
    universe: List[Dict[str, Any]] = []

    for raw in mapping:
        m = _coerce_entry(raw)
        tkr = m.get("ticker", "")
        if not tkr or tkr in ("ETH","DOT"):  # excluded from rotation
            continue

        # try binance
        md = {}
        bsym = best_binance_symbol(m)
        if bsym:
            md = binance_24h_and_book(bsym)

        # fallback coinbase
        if not md.get("ok"):
            prod = best_coinbase_product(m)
            if prod:
                md = coinbase_24h_and_book(prod)

        if not md.get("ok"):
            continue

        spr = spread_pct(md["bid"], md["ask"])
        if md["volume_usd"] >= 8_000_000 and spr <= 0.005:
            universe.append({"ticker": tkr, "source": md.get("src",""),
                             "symbol": (bsym if md.get("src")=="binance" else best_coinbase_product(m)),
                             "price": md["price"], "bid": md["bid"], "ask": md["ask"],
                             "spread": spr, "vol_usd": md["volume_usd"]})

    snapshot["universe_count"] = len(universe)

    # 3 & 4) Signals (only supported with Binance klines for now)
    scored: List[Dict[str, Any]] = []
    for u in universe:
        if u["source"] != "binance":  # we only have the 1m klines adapter for Binance here
            continue
        bars = binance_klines_1m(best_binance_symbol({"ticker": u["ticker"]}), limit=120)
        if not bars: continue
        br = aggressive_breakout(bars)
        if not br: continue
        mp = micro_pullback_ok(bars)
        if not mp: continue
        atr1m = atr_from_klines(bars, period=14)
        if atr1m <= 0: continue
        score = br["rvol"] * (1.0 + br["pct"])
        scored.append({"ticker": u["ticker"], "symbol": u["symbol"],
                       "entry": mp["entry"], "pullback_low": mp["pullback_low"],
                       "atr1m": atr1m, "rvol": br["rvol"], "pct": br["pct"], "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)

    if not scored:
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        print("Hold and wait."); _write_signal("C", "Hold and wait."); return

    top = scored[0]
    entry = float(top["entry"]); stop = float(top["pullback_low"]); atr = float(top["atr1m"])
    t1 = entry + 0.8*atr; t2 = entry + 1.5*atr
    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    snapshot["candidates"] = [top]
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))

    sell_plan = "Sell weakest non-ETH, non-DOT on support break if cash needed."
    text = (
        f"• {top['ticker']} + {top['ticker']}\n"
        f"• Entry price: {entry:.6f}\n"
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R\n"
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min\n"
        f"• What to sell from portfolio (excluding ETH, DOT): {sell_plan}\n"
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    )
    print(text); _write_signal("B", text)

if __name__ == "__main__":
    main()