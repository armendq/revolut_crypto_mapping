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
DEBUG_LOG = []

VOL_USD_MIN = 8_000_000         # universe volume threshold
MAX_SPREAD = 0.005               # ≤ 0.5%
RISK_PCT = 0.012                 # 1.2% risk
STRONG_ALLOC = 0.60
WEAK_ALLOC = 0.30
EQUITY_FLOOR = 36500.0

# ---------- small utilities ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _as_float(x: Any) -> float:
    """Convert pandas objects / scalars to float robustly."""
    if hasattr(x, "iloc"):
        return float(x.iloc[-1])
    return float(x)

def _http_json(url: str, timeout: int = 20, retries: int = 2, backoff: float = 0.25):
    """GET JSON with tiny retry + debug logging."""
    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.0"})
        try:
            t0 = perf_counter()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            DEBUG_LOG.append({"stage": "http", "url": url, "ms": int((perf_counter()-t0)*1000), "ok": True})
            return data
        except Exception as e:
            last_err = str(e)
            DEBUG_LOG.append({"stage": "http", "url": url, "ok": False, "err": last_err})
            time.sleep(backoff)
    return None

# ---------- Exchange adapters ----------
def binance_24h_and_book(symbol_usdt: str) -> Dict[str, Any]:
    """
    Returns dict: {'ok', 'price', 'volume_usd', 'bid', 'ask'} (ok=False on failure)
    """
    try:
        t = _http_json("https://api.binance.com/api/v3/ticker/24hr?symbol=" + symbol_usdt)
        if not t:
            return {"ok": False}
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last

        ob = _http_json("https://api.binance.com/api/v3/depth?symbol=" + symbol_usdt + "&limit=5")
        if not ob:
            return {"ok": False}
        bid = float(ob["bids"][0][0])
        ask = float(ob["asks"][0][0])

        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
    except Exception as e:
        DEBUG_LOG.append({"stage": "24h+book", "symbol": symbol_usdt, "err": str(e)})
        return {"ok": False}

def binance_klines_1m(symbol_usdt: str, limit: int = 120) -> List[Dict[str, float]]:
    """Return list of dict bars with keys ts,o,h,l,c,v; [] on failure."""
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
    ]
    for u in urls:
        arr = _http_json(u)
        if isinstance(arr, list) and arr:
            try:
                return [
                    {"ts": x[0], "o": float(x[1]), "h": float(x[2]), "l": float(x[3]),
                     "c": float(x[4]), "v": float(x[5])}
                    for x in arr
                ]
            except Exception:
                continue
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
    out = []
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

# ---------- MAIN ----------
def main():
    t_run0 = perf_counter()
    DEBUG_LOG.append({"stage": "run", "event": "start", "ts": _now_iso()})

    # Inputs (GH env defaults shown)
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
        "candidates": []
    }

    # Capital preservation guard
    buffer_ok = (equity - EQUITY_FLOOR) >= 1000.0
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=37500"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type": "A", "text": "Raise cash now: halt new entries; exit weakest on breaks."}, indent=2))
        RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t_run0)*1000), "time": _now_iso()}, indent=2))
        ARTIFACTS.joinpath("debug_scan.json").write_text(json.dumps(DEBUG_LOG, indent=2))
        print("Raise cash now.")
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
        # Output C — hold
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type": "C", "text": "Hold and wait."}, indent=2))
        print("Hold and wait. (No qualified candidates.)")
    else:
        top = candidates[0]
        entry = top["entry"]
        stop = top["stop"]
        atr = top["atr1m"]
        t1 = entry + 0.8 * atr
        t2 = entry + 1.5 * atr
        buy_usd = position_size(entry, stop, equity, cash, strong)

        # Write human-readable trade plan to data/signals.json
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
        print("\n".join(plan_lines))

    # 5) run stats + debug
    RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t_run0)*1000), "time": _now_iso()}, indent=2))
    ARTIFACTS.joinpath("debug_scan.json").write_text(json.dumps(DEBUG_LOG, indent=2))

if __name__ == "__main__":
    main()