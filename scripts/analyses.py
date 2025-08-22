# scripts/analyses.py
import os
import json
import time
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from time import perf_counter

import pandas as pd

# ---- local helpers from your repo ----
from scripts.marketdata import get_btc_5m_klines, ema, vwap

# ---------- Paths / constants ----------
ARTIFACTS = Path("artifacts")
DATA = Path("data")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SNAP_PATH = ARTIFACTS / "market_snapshot.json"
SIGNAL_PATH = DATA / "signals.json"
RUN_STATS = ARTIFACTS / "run_stats.json"
DEBUG_PATH = ARTIFACTS / "debug_scan.json"
DEBUG_LOG: List[Dict[str, Any]] = []

VOL_USD_MIN = 8_000_000         # universe volume threshold
MAX_SPREAD = 0.005               # ≤ 0.5%
RISK_PCT = 0.012                 # 1.2% risk
STRONG_ALLOC = 0.60
WEAK_ALLOC = 0.30
EQUITY_FLOOR = 36500.0

# ----- Binance mirror bases to mitigate regional blocking (HTTP 451) -----
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api4.binance.com",
    "https://api-gcp.binance.com",
    "https://data-api.binance.vision",
]
blocked_451 = False  # set True if we only see 451s (or all Binance bases fail)

# ---------- small utilities ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _as_float(x: Any) -> float:
    """Convert pandas objects / scalars to float robustly."""
    if hasattr(x, "iloc"):
        return float(x.iloc[-1])
    return float(x)

def _http_json(url: str, timeout: int = 20) -> Optional[Any]:
    """
    GET JSON; return None on error. Logs timing and errors to DEBUG_LOG.
    (Retries are handled by trying multiple mirror bases rather than re-calling the same host.)
    """
    req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.0"})
    try:
        t0 = perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        DEBUG_LOG.append({"stage": "http", "url": url, "ms": int((perf_counter()-t0)*1000), "ok": True})
        return data
    except Exception as e:
        msg = str(e)
        DEBUG_LOG.append({"stage": "http", "url": url, "ok": False, "err": msg})
        return None

def _saw_451_recent(n: int = 20) -> bool:
    """Scan the last N http logs for HTTP 451 errors."""
    for e in DEBUG_LOG[-n:]:
        if isinstance(e, dict) and "err" in e and "451" in str(e["err"]):
            return True
    return False

# ---------- Exchange adapters ----------
def binance_24h_and_book(symbol_usdt: str) -> Dict[str, Any]:
    """
    Returns dict: {'ok', 'price', 'volume_usd', 'bid', 'ask'} (ok=False on failure)
    Tries several Binance base URLs to avoid regional blocks.
    """
    global blocked_451
    for base in BINANCE_BASES:
        t_url = f"{base}/api/v3/ticker/24hr?symbol={symbol_usdt}"
        d_url = f"{base}/api/v3/depth?symbol={symbol_usdt}&limit=5"
        t = _http_json(t_url)
        d = _http_json(d_url) if t else None
        if t and d:
            try:
                last = float(t["lastPrice"])
                vol_base = float(t["volume"])
                vol_usd = vol_base * last
                bid = float(d["bids"][0][0])
                ask = float(d["asks"][0][0])
                return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
            except Exception as e:
                DEBUG_LOG.append({"stage": "24h+book-parse", "symbol": symbol_usdt, "err": str(e)})
                continue
    # all bases failed -> mark possible 451
    if _saw_451_recent():
        blocked_451 = True
    return {"ok": False}

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict[str, float]]:
    """Return list of dict bars with keys ts,o,h,l,c,v; [] on failure. Tries multiple bases."""
    global blocked_451
    for base in BINANCE_BASES:
        u = f"{base}/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}"
        arr = _http_json(u)
        if isinstance(arr, list) and arr:
            try:
                return [
                    {"ts": x[0], "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
                     "c": float(x[4]), "v": float(x[5])}
                    for x in arr
                ]
            except Exception as e:
                DEBUG_LOG.append({"stage": "klines-parse", "symbol": symbol_usdt, "err": str(e)})
                continue
    if _saw_451_recent():
        blocked_451 = True
    return []

# ---------- TA helpers ----------
def true_range(h: float, l: float, pc: float) -> float:
    return max(h - l, abs(h - pc), abs(l - pc))

def atr_from_klines(bars: List[Dict[str, float]], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period + 1):
        h = bars[-i]["h"]
        l = bars[-i]["l"]
        pc = bars[-i - 1]["c"]
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

# ---------- Signal rules ----------
def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    1-min bar close +1.8%..+4.0% vs prev close AND RVOL >= 4 vs last 15 bars' median,
    AND last close above the last 15-bar high (follow-through).
    """
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]
    prev = bars_1m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    if pct < 0.018 or pct > 0.04:
        return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]])
    rvol = (last["v"] / vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0005:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    After breakout: last bar has small fade & small dip (<= 0.6% of high).
    Entry at last close, stop at that bar's low.
    """
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] == 0 or last["c"] == 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    micro = (last["h"] - last["l"]) / last["h"]
    if fade <= 0.006 and micro <= 0.006:
        return {"entry": last["c"], "stop": last["l"]}
    return None

# ---------- Regime ----------
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
        # very defensive fallback: treat as not-ok
        return {"ok": False, "reason": "calc-error"}
    ok = (last > vw) and (last > e9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-or-ema",
            "last": last, "vwap": vw, "ema9": e9}

# ---------- Universe / mapping ----------
def load_revolut_mapping() -> List[Dict[str, Any]]:
    with open(DATA / "revolut_mapping.json", "r", encoding="utf-8") as f:
        obj = json.load(f)
    # ensure list of dicts
    out: List[Dict[str, Any]] = []
    for m in obj:
        if isinstance(m, dict):
            out.append(m)
        elif isinstance(m, str):
            out.append({"ticker": m})
    return out

def best_binance_symbol(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("binance"):
        return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# ---------- Sizing ----------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = STRONG_ALLOC if strong_regime else WEAK_ALLOC
    usd_cap = equity * cap
    return max(0.0, min(usd, usd_cap, cash))

# ---------- helpers to persist outputs ----------
def _write_all(snapshot: Dict[str, Any],
               signal_type: str,
               signal_text: str,
               t_start: float) -> None:
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
    SIGNAL_PATH.write_text(json.dumps({"type": signal_type, "text": signal_text}, indent=2))
    RUN_STATS.write_text(
        json.dumps({"elapsed_ms": int((perf_counter() - t_start) * 1000),
                    "time": _now_iso(),
                    "universe_count": snapshot.get("universe_count", 0),
                    "candidates_count": len(snapshot.get("candidates", [])),
                    "regime_ok": bool(snapshot.get("regime", {}).get("ok", False))}, indent=2)
    )
    DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))

# ---------- MAIN ----------
def main():
    t_run0 = perf_counter()
    DEBUG_LOG.append({"stage": "run", "event": "start", "ts": _now_iso()})

    # Inputs (GH env defaults shown)
    equity = float(os.getenv("EQUITY", "41000") or "41000")
    cash = float(os.getenv("CASH", "32000") or "32000")

    snapshot: Dict[str, Any] = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": []
    }

    # Capital preservation guard
    buffer_ok = (equity - EQUITY_FLOOR) >= 1000.0
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        _write_all(snapshot, "A",
                   "Raise cash now: halt new entries; exit weakest on breaks.",
                   t_run0)
        print("Raise cash now.")
        return

    # ---- Quick probe: is Binance reachable at all? (handles HTTP 451 gracefully) ----
    _ = binance_24h_and_book("BTCUSDT")
    if blocked_451:
        snapshot["regime"] = {"ok": False, "reason": "binance-451-block"}
        snapshot["universe_count"] = 0
        _write_all(snapshot, "C",
                   "Data source blocked (HTTP 451). Hold and wait.",
                   t_run0)
        print("Hold and wait. (Binance HTTP 451 block detected.)")
        return

    # 1) Regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong = bool(regime.get("ok", False))

    # 2) Universe
    mapping = load_revolut_mapping()
    universe = []
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if tkr in ("ETH", "DOT"):
            continue  # excluded from rotation
        sym = best_binance_symbol(m)
        if not sym:
            continue
        md = binance_24h_and_book(sym)
        if not md.get("ok"):
            continue
        spr = spread_pct(md["bid"], md["ask"])
        if md["volume_usd"] >= VOL_USD_MIN and spr <= MAX_SPREAD:
            universe.append({
                "ticker": tkr,
                "symbol": sym,
                "price": md["price"],
                "spread": spr,
                "vol_usd": md["volume_usd"]
            })

    snapshot["universe_count"] = len(universe)

    # 3) Signals (breakout + micro pullback), then score by RVOL*(1+pct)
    candidates = []
    for u in universe:
        bars = binance_klines_1m(u["symbol"], limit=120)
        if not bars:
            continue
        br = aggressive_breakout(bars)
        if not br:
            continue
        pb = micro_pullback_ok(bars)
        if not pb:
            continue
        atr1m = atr_from_klines(bars, period=14)
        if atr1m <= 0:
            continue
        score = br["rvol"] * (1.0 + br["pct"])
        candidates.append({
            "ticker": u["ticker"],
            "symbol": u["symbol"],
            "entry": pb["entry"],
            "stop": pb["stop"],
            "atr1m": atr1m,
            "rvol": br["rvol"],
            "pct": br["pct"],
            "score": score
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    snapshot["candidates"] = candidates[:3]  # keep a short list in snapshot

    # 4) Outputs
    if not strong or not candidates:
        _write_all(snapshot, "C", "Hold and wait.", t_run0)
        print("Hold and wait. (No qualified candidates.)")
        return

    top = candidates[0]
    entry = top["entry"]
    stop = top["stop"]
    atr = top["atr1m"]
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr
    buy_usd = position_size(entry, stop, equity, cash, strong)

    # Human-readable trade plan
    plan_lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below {stop:.6f} or stall > 5 min",
        "• What to sell from portfolio (excluding ETH, DOT): Sell weakest on support break if cash needed.",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    ]
    _write_all(snapshot, "B", "\n".join(plan_lines), t_run0)
    print("\n".join(plan_lines))

if __name__ == "__main__":
    main()