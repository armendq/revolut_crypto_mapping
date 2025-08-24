# scripts/analyses.py
import os
import json
import time
import statistics
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from time import perf_counter

import pandas as pd

from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------- paths ----------
ARTIFACTS = Path("artifacts")
DATA = Path("data")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNAL_PATH = DATA / "signals.json"
RUN_STATS = ARTIFACTS / "run_stats.json"
DEBUG_PATH = ARTIFACTS / "debug_scan.json"
DEBUG_LOG: List[Dict[str, Any]] = []

# ---------- thresholds (relaxed) ----------
VOL_USD_MIN = 5_000_000          # was 8M
MAX_SPREAD = 0.008               # was 0.5% → 0.8%
RISK_PCT = 0.012
STRONG_ALLOC = 0.60
WEAK_ALLOC = 0.30
EQUITY_FLOOR = 36_500.0

# ---------- API bases ----------
BINANCE_BASES = [
    "https://data-api.binance.vision",
]
COINBASE_BASE = "https://api.exchange.coinbase.com"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

blocked_451 = False

# ---------- utils ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _as_float(x: Any) -> float:
    if hasattr(x, "iloc"):
        return float(x.iloc[-1])
    return float(x)

def _http_json(url: str, timeout: int = 20) -> Optional[Any]:
    global blocked_451
    req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.0"})
    t0 = perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            data = json.loads(raw)
        DEBUG_LOG.append({"stage": "http", "url": url, "ms": int((perf_counter()-t0)*1000), "ok": True})
        return data
    except urllib.error.HTTPError as he:
        if he.code == 451:
            blocked_451 = True
        DEBUG_LOG.append({"stage": "http", "url": url, "ok": False, "err": f"HTTP {he.code}"})
        return None
    except Exception as e:
        DEBUG_LOG.append({"stage": "http", "url": url, "ok": False, "err": str(e)})
        return None

# ---------- CoinGecko ----------
def cg_search_id(symbol: str) -> Optional[str]:
    q = symbol.upper()
    data = _http_json(f"{COINGECKO_BASE}/search?query={q}")
    if not data or "coins" not in data:
        return None
    for c in data["coins"]:
        if str(c.get("symbol","")).upper() == q:
            return c.get("id")
    return None

def cg_price_and_vol_usd(symbol: str) -> Optional[Dict[str, float]]:
    cid = cg_search_id(symbol)
    if not cid:
        return None
    data = _http_json(f"{COINGECKO_BASE}/simple/price?ids={cid}&vs_currencies=usd&include_24hr_vol=true")
    if not data or cid not in data:
        return None
    try:
        d = data[cid]
        return {"price": float(d.get("usd")), "vol_usd": float(d.get("usd_24h_vol", 0.0))}
    except Exception:
        return None

# ---------- Coinbase ----------
def cb_product_id(symbol: str) -> str:
    return f"{symbol.upper()}-USD"

def cb_orderbook_best(symbol: str) -> Optional[Dict[str, float]]:
    pid = cb_product_id(symbol)
    data = _http_json(f"{COINBASE_BASE}/products/{pid}/book?level=1")
    if not data or "bids" not in data or "asks" not in data:
        return None
    try:
        bid = float(data["bids"][0][0]); ask = float(data["asks"][0][0])
        return {"bid": bid, "ask": ask}
    except Exception:
        return None

def cb_24h_stats(symbol: str) -> Optional[Dict[str, float]]:
    pid = cb_product_id(symbol)
    data = _http_json(f"{COINBASE_BASE}/products/{pid}/stats")
    if not data:
        return None
    try:
        last = float(data.get("last", 0.0))
        vol_base = float(data.get("volume", 0.0))
        return {"price": last, "vol_usd": vol_base * last}
    except Exception:
        return None

def cb_klines_1m(symbol: str, limit: int = 120) -> List[Dict[str, float]]:
    pid = cb_product_id(symbol)
    data = _http_json(f"{COINBASE_BASE}/products/{pid}/candles?granularity=60")
    if not isinstance(data, list) or not data:
        return []
    try:
        rows = sorted(data, key=lambda r: r[0])[-limit:]
        return [{"ts": t*1000, "o": float(op), "h": float(hi), "l": float(lo), "c": float(cl), "v": float(vol)}
                for t, lo, hi, op, cl, vol in rows]
    except Exception:
        return []

# ---------- Binance ----------
def binance_24h_and_book(symbol_usdt: str) -> Optional[Dict[str, float]]:
    last_err = None
    for base in BINANCE_BASES:
        t = _http_json(f"{base}/api/v3/ticker/24hr?symbol={symbol_usdt}")
        d = _http_json(f"{base}/api/v3/depth?symbol={symbol_usdt}&limit=5") if t else None
        if t and d:
            try:
                last = float(t["lastPrice"])
                vol_usd = float(t["volume"]) * last
                bid = float(d["bids"][0][0]); ask = float(d["asks"][0][0])
                return {"price": last, "vol_usd": vol_usd, "bid": bid, "ask": ask}
            except Exception as e:
                last_err = str(e)
                continue
    if last_err:
        DEBUG_LOG.append({"stage": "binance_24h_and_book", "err": last_err})
    return None

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict[str, float]]:
    for base in BINANCE_BASES:
        arr = _http_json(f"{base}/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}")
        if isinstance(arr, list) and arr:
            try:
                return [{"ts": x[0], "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
                         "c": float(x[4]), "v": float(x[5])} for x in arr]
            except Exception:
                continue
    return []

# ---------- TA ----------
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

def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

# ---------- signal rules (relaxed) ----------
def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Relaxed: +1.0%..+5.5% vs prev close AND RVOL >= 2.5 AND new 15-bar high.
    """
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]; prev = bars_1m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    if pct < 0.010 or pct > 0.055:
        return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]])
    rvol = (last["v"] / vol_med) if vol_med > 0 else 0.0
    if rvol < 2.5:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0001:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Relaxed: fade <= 1.2% and intrabar range <= 1.5% of high.
    """
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] == 0 or last["c"] == 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.012 and micro <= 0.015:
        return {"entry": last["c"], "stop": last["l"]}
    return None

# ---------- regime ----------
def check_regime() -> Dict[str, Any]:
    bars = get_btc_5m_klines()
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        return {"ok": False, "reason": "no-btc-5m"}
    close = bars["close"]
    try:
        last = _as_float(close.iloc[-1])
        vw = _as_float(vwap(bars))
        e9 = _as_float(ema(close, span=9))
    except Exception:
        return {"ok": False, "reason": "calc-error"}
    ok = (last > vw) and (last > e9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-or-ema",
            "last": last, "vwap": vw, "ema9": e9}

# ---------- mapping ----------
def load_revolut_mapping() -> List[Dict[str, Any]]:
    path = DATA / "revolut_mapping.json"
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for m in obj:
            if isinstance(m, dict):
                out.append(m)
            elif isinstance(m, str):
                out.append({"ticker": m})
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                vv = dict(v); vv.setdefault("ticker", k); out.append(vv)
            else:
                out.append({"ticker": k})
    return out

def best_binance_symbol(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# ---------- data fetch ----------
def fetch_meta_for_symbol(ticker: str, binance_sym: Optional[str]) -> Optional[Dict[str, float]]:
    # 1) Binance
    if binance_sym:
        b = binance_24h_and_book(binance_sym)
        if b:
            spr = spread_pct(b["bid"], b["ask"])
            return {"price": b["price"], "vol_usd": b["vol_usd"], "bid": b["bid"], "ask": b["ask"],
                    "spread": spr, "src": "binance"}
    # 2) Coinbase
    cb_stats = cb_24h_stats(ticker)
    cb_book = cb_orderbook_best(ticker)
    if cb_stats and cb_book:
        spr = spread_pct(cb_book["bid"], cb_book["ask"])
        return {"price": cb_stats["price"], "vol_usd": cb_stats["vol_usd"],
                "bid": cb_book["bid"], "ask": cb_book["ask"], "spread": spr, "src": "coinbase"}
    # 3) CoinGecko
    cg = cg_price_and_vol_usd(ticker)
    if cg:
        return {"price": cg["price"], "vol_usd": cg["vol_usd"], "bid": None, "ask": None, "spread": None, "src": "coingecko"}
    return None

def get_1m_bars_any(ticker: str, binance_sym: Optional[str], limit: int = 120) -> List[Dict[str, float]]:
    if binance_sym:
        b = binance_klines_1m(binance_sym, limit=limit)
        if b:
            return b
    return cb_klines_1m(ticker, limit=limit)

# ---------- sizing ----------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = STRONG_ALLOC if strong_regime else WEAK_ALLOC
    usd_cap = equity * cap
    return max(0.0, min(usd, usd_cap, cash))

# ---------- MAIN ----------
def main():
    t_run0 = perf_counter()
    DEBUG_LOG.append({"stage": "run", "event": "start", "ts": _now_iso()})

    equity = float(os.getenv("EQUITY", "41000") or "41000")
    cash = float(os.getenv("CASH", "32000") or "32000")

    snapshot = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": [],
        "reject_stats": {}
    }

    # capital guard
    buffer_ok = (equity - EQUITY_FLOOR) >= 1000.0
    if not buffer_ok or equity <= 37_500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."}, indent=2))
        RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t_run0)*1000), "time": _now_iso()}, indent=2))
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        print("Raise cash now.")
        return

    # 1) regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong = bool(regime.get("ok", False))

    # 2) universe
    mapping = load_revolut_mapping()
    universe = []
    rej_counts = {"meta":0, "no_spread":0, "vol":0, "spread":0, "bars":0, "br":0, "pb":0, "atr":0}
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if not tkr:
            continue
        if tkr in ("ETH", "DOT"):  # excluded from rotation
            continue
        sym = best_binance_symbol(m)
        meta = fetch_meta_for_symbol(tkr, sym)
        if not meta:
            rej_counts["meta"] += 1
            continue
        spr = meta.get("spread")
        vol_usd = float(meta.get("vol_usd") or 0.0)
        if spr is None:
            rej_counts["no_spread"] += 1
            DEBUG_LOG.append({"stage": "universe-skip", "ticker": tkr, "reason": "no-spread"})
            continue
        if vol_usd < VOL_USD_MIN:
            rej_counts["vol"] += 1
            continue
        if spr > MAX_SPREAD:
            rej_counts["spread"] += 1
            continue
        universe.append({
            "ticker": tkr,
            "symbol": sym or cb_product_id(tkr),
            "price": float(meta["price"]),
            "bid": float(meta["bid"]),
            "ask": float(meta["ask"]),
            "spread": spr,
            "vol_usd": vol_usd,
            "src": meta.get("src")
        })

    snapshot["universe_count"] = len(universe)

    # 3) signals
    candidates = []
    for u in universe:
        bars = get_1m_bars_any(u["ticker"], u.get("symbol") if u.get("src") == "binance" else best_binance_symbol({"ticker": u["ticker"]}), limit=120)
        if not bars:
            rej_counts["bars"] += 1
            continue
        br = aggressive_breakout(bars)
        if not br:
            rej_counts["br"] += 1
            continue
        pb = micro_pullback_ok(bars)
        if not pb:
            rej_counts["pb"] += 1
            continue
        atr1m = atr_from_klines(bars, period=14)
        if atr1m <= 0:
            rej_counts["atr"] += 1
            continue
        score = br["rvol"] * (1.0 + br["pct"])
        candidates.append({
            "ticker": u["ticker"],
            "symbol": u["symbol"],
            "entry": float(pb["entry"]),
            "stop": float(pb["stop"]),
            "atr1m": float(atr1m),
            "rvol": float(br["rvol"]),
            "pct": float(br["pct"]),
            "score": float(score)
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    snapshot["candidates"] = candidates[:3]
    snapshot["reject_stats"] = rej_counts

    # 4) outputs
    if not strong or not candidates:
        if blocked_451:
            snapshot["regime"] = {"ok": False, "reason": "binance-451-block"}
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type": "C", "text": "Hold and wait."}, indent=2))
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t_run0)*1000), "time": _now_iso()}, indent=2))
        print("Hold and wait. (No qualified candidates.)")
        return

    top = candidates[0]
    entry = top["entry"]; stop = top["stop"]; atr = top["atr1m"]
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr
    buy_usd = position_size(entry, stop, equity, cash, strong)

    plan_lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below {stop:.6f} or stall > 5 min",
        "• What to sell from portfolio (excluding ETH, DOT): Sell weakest on support break if cash needed.",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    ]
    SIGNAL_PATH.write_text(json.dumps({"type": "B", "text": "\n".join(plan_lines)}, indent=2))

    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
    DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
    RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t_run0)*1000), "time": _now_iso()}, indent=2))
    print("\n".join(plan_lines))

if __name__ == "__main__":
    main()