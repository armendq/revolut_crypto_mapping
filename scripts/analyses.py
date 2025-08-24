# scripts/analyses.py
import os
import json
import statistics
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from time import perf_counter

import pandas as pd

# ---- local helpers from your repo ----
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

# ---------- config (relaxed, two-lane) ----------
# Read floor from env, default 40k per your update
EQUITY_FLOOR = float(os.getenv("EQUITY_FLOOR", "40000"))

# Lane A (orderbook-grade)
VOL_USD_MIN_A = 3_000_000        # 24h USD volume threshold
MAX_SPREAD_A  = 0.012            # 1.2%

# Lane B (CG-only momentum “exception lane”)
VOL_USD_MIN_B = 15_000_000       # higher vol bar for CG-only
EXC_1H_MIN    = 0.08             # +8% 1h
EXC_24H_MIN   = 0.20             # +20% 24h

# Breakout/micro-pullback
BR_MIN = 0.008                   # +0.8% vs prev close
BR_MAX = 0.12                    # +12% vs prev close
RVOL_MIN = 2.0                   # relaxed
FADE_MAX = 0.012                 # 1.2%
RANGE_MAX = 0.015                # 1.5%

RISK_PCT = 0.012                 # 1.2% risk
STRONG_ALLOC = 0.60
WEAK_ALLOC   = 0.30

# APIs
BINANCE_BASES = ["https://data-api.binance.vision"]
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
    req = urllib.request.Request(url, headers={"User-Agent": "rev-scan/1.1"})
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

# ---------- CoinGecko helpers ----------
def cg_search_id(symbol: str) -> Optional[str]:
    q = symbol.upper()
    data = _http_json(f"{COINGECKO_BASE}/search?query={q}")
    if not data or "coins" not in data:
        return None
    for c in data["coins"]:
        if str(c.get("symbol","")).upper() == q:
            return c.get("id")
    return None

def cg_simple_price_vol(cid: str) -> Optional[Dict[str, float]]:
    d = _http_json(f"{COINGECKO_BASE}/simple/price?ids={cid}&vs_currencies=usd&include_24hr_vol=true")
    if not d or cid not in d: return None
    try:
        x = d[cid]
        return {"price": float(x.get("usd")), "vol_usd": float(x.get("usd_24h_vol", 0.0))}
    except Exception:
        return None

def cg_markets(cid: str) -> Optional[Dict[str, float]]:
    d = _http_json(f"{COINGECKO_BASE}/coins/markets?vs_currency=usd&ids={cid}&price_change_percentage=1h,24h")
    if not isinstance(d, list) or not d: return None
    try:
        x = d[0]
        return {
            "pc_1h": float(x.get("price_change_percentage_1h_in_currency") or 0.0)/100.0,
            "pc_24h": float(x.get("price_change_percentage_24h_in_currency") or 0.0)/100.0,
            "current_price": float(x.get("current_price") or 0.0),
        }
    except Exception:
        return None

def cg_minute_prices(cid: str) -> List[List[float]]:
    d = _http_json(f"{COINGECKO_BASE}/coins/{cid}/market_chart?vs_currency=usd&days=1&interval=minutely")
    if not d or "prices" not in d: return []
    return d["prices"]  # [ [ms, price], ... ]

def cg_1m_bars_from_chart(cid: str, limit: int = 120) -> List[Dict[str, float]]:
    pts = cg_minute_prices(cid)
    if not pts: return []
    # build pseudo-1m candles
    bars: List[Dict[str, float]] = []
    for i in range(max(1, len(pts) - limit), len(pts)):
        t, c = pts[i]
        prev_c = bars[-1]["c"] if bars else c
        o = prev_c
        h = max(o, c)
        l = min(o, c)
        bars.append({"ts": int(t), "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": 0.0})
    return bars[-limit:]

# ---------- Coinbase ----------
def cb_product_id(symbol: str) -> str:
    return f"{symbol.upper()}-USD"

def cb_orderbook_best(symbol: str) -> Optional[Dict[str, float]]:
    pid = cb_product_id(symbol)
    data = _http_json(f"{COINBASE_BASE}/products/{pid}/book?level=1")
    if not data or "bids" not in data or "asks" not in data:
        return None
    try:
        bid = float(data["bids")[0][0])  # typo guard below
    except Exception:
        try:
            bid = float(data["bids"][0][0]); ask = float(data["asks"][0][0])
            return {"bid": bid, "ask": ask}
        except Exception:
            return None
    ask = float(data["asks"][0][0])
    return {"bid": bid, "ask": ask}

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

# ---------- TA rules ----------
def aggressive_breakout(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]; prev = bars_1m[-2]
    pct = (last["c"] / prev["c"]) - 1.0
    if pct < BR_MIN or pct > BR_MAX:
        return None
    vol_med = median([b["v"] for b in bars_1m[-16:-1]]) if any(b["v"] for b in bars_1m[-16:-1]) else 0.0
    rvol = (last["v"] / vol_med) if vol_med > 0 else 3.0  # default proxy if no volume
    if rvol < RVOL_MIN:
        return None
    hh15 = max(b["h"] for b in bars_1m[-15:])
    if last["c"] <= hh15 * 1.0001:
        return None
    return {"pct": pct, "rvol": rvol, "hh15": hh15}

def micro_pullback_ok(bars_1m: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last["h"] == 0 or last["c"] == 0:
        return None
    fade = (last["h"] - last["c"]) / last["h"]
    intra = (last["h"] - last["l"]) / last["h"]
    if fade <= FADE_MAX and intra <= RANGE_MAX:
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
    # Relaxed gate: either VWAP or EMA9
    ok = (last > vw) or (last > e9)
    return {"ok": ok, "reason": "" if ok else "btc-below-vwap-and-ema", "last": last, "vwap": vw, "ema9": e9}

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
    if entry.get("binance"): return entry["binance"]
    t = (entry.get("ticker") or "").upper()
    return f"{t}USDT" if t else None

# ---------- data fetch combo ----------
def fetch_meta_for_symbol(ticker: str, binance_sym: Optional[str]) -> Dict[str, Any]:
    """
    Returns meta dict with keys:
      src, price, vol_usd, bid, ask, spread, cid, pc_1h, pc_24h
    """
    cid = cg_search_id(ticker) or ""
    # A) Binance
    if binance_sym:
        b = binance_24h_and_book(binance_sym)
        if b:
            spr = spread_pct(b["bid"], b["ask"])
            return {"src":"binance","price":b["price"],"vol_usd":b["vol_usd"],"bid":b["bid"],"ask":b["ask"],
                    "spread":spr,"cid":cid,"pc_1h":None,"pc_24h":None}
    # B) Coinbase
    cb_stats = cb_24h_stats(ticker)
    cb_book  = cb_orderbook_best(ticker)
    if cb_stats and cb_book:
        spr = spread_pct(cb_book["bid"], cb_book["ask"])
        return {"src":"coinbase","price":cb_stats["price"],"vol_usd":cb_stats["vol_usd"],
                "bid":cb_book["bid"],"ask":cb_book["ask"],"spread":spr,"cid":cid,"pc_1h":None,"pc_24h":None}
    # C) CoinGecko only
    if cid:
        sp = cg_simple_price_vol(cid)
        mk = cg_markets(cid) or {}
        if sp:
            return {"src":"coingecko","price":sp["price"],"vol_usd":sp["vol_usd"],"bid":None,"ask":None,"spread":None,
                    "cid":cid,"pc_1h":mk.get("pc_1h"),"pc_24h":mk.get("pc_24h")}
    return {"src":"none","price":0.0,"vol_usd":0.0,"bid":None,"ask":None,"spread":None,"cid":cid,"pc_1h":None,"pc_24h":None}

def get_1m_bars_any(ticker: str, meta: Dict[str, Any], binance_sym: Optional[str], limit: int = 120) -> List[Dict[str, float]]:
    if meta["src"] == "binance" and binance_sym:
        b = binance_klines_1m(binance_sym, limit=limit)
        if b: return b
    if meta["src"] == "coinbase":
        b = cb_klines_1m(ticker, limit=limit)
        if b: return b
    # CG fallback
    if meta.get("cid"):
        return cg_1m_bars_from_chart(meta["cid"], limit=limit)
    return []

# ---------- sizing ----------
def position_size(entry: float, stop: float, equity: float, cash: float, strong_regime: bool, lane: str) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-7)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = STRONG_ALLOC if strong_regime else WEAK_ALLOC
    # Tighter cap for CG-only exception lane
    per_trade_cap = cap * equity * (0.5 if lane == "B" else 1.0)
    return max(0.0, min(usd, per_trade_cap, cash))

# ---------- MAIN ----------
def main():
    t0 = perf_counter()
    DEBUG_LOG.append({"stage":"run","event":"start","ts":_now_iso()})

    equity = float(os.getenv("EQUITY", "41000") or "41000")
    cash   = float(os.getenv("CASH",   "32000") or "32000")

    snapshot = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": [],
        "reject_stats": {"A_meta":0,"A_vol":0,"A_spread":0,"A_rules":0,"B_meta":0,"B_vol":0,"B_momentum":0,"bars":0}
    }

    # floor guard
    buffer_ok = (equity - EQUITY_FLOOR) >= 1000.0
    if not buffer_ok or equity <= EQUITY_FLOOR:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "buffer<1000_over_floor" if not buffer_ok else "equity<=floor"
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type":"A","text":"Raise cash now: halt new entries; exit weakest on breaks."}, indent=2))
        RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t0)*1000), "time": _now_iso()}, indent=2))
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        print("Raise cash now.")
        return

    # regime
    regime = check_regime()
    snapshot["regime"] = regime
    strong = bool(regime.get("ok", False))

    # mapping
    mapping = load_revolut_mapping()
    orderbook_lane = []
    exception_lane = []
    for m in mapping:
        tkr = (m.get("ticker") or "").upper()
        if not tkr: continue
        if tkr in ("ETH","DOT"):  # staked / not rotated
            continue
        sym = best_binance_symbol(m)
        meta = fetch_meta_for_symbol(tkr, sym)
        src = meta["src"]

        # Lane A: orderbook-grade
        if src in ("binance","coinbase"):
            if meta["vol_usd"] < VOL_USD_MIN_A:
                snapshot["reject_stats"]["A_vol"] += 1; continue
            if meta.get("spread", 1.0) > MAX_SPREAD_A:
                snapshot["reject_stats"]["A_spread"] += 1; continue
            bars = get_1m_bars_any(tkr, meta, sym, limit=120)
            if not bars:
                snapshot["reject_stats"]["bars"] += 1; continue
            br = aggressive_breakout(bars)
            pb = micro_pullback_ok(bars) if br else None
            if not (br and pb):
                snapshot["reject_stats"]["A_rules"] += 1; continue
            atr1m = atr_from_klines(bars, period=14)
            if atr1m <= 0: snapshot["reject_stats"]["A_rules"] += 1; continue
            score = br["rvol"] * (1.0 + br["pct"])
            orderbook_lane.append({
                "lane":"A","ticker":tkr,"symbol":sym or cb_product_id(tkr),
                "entry": float(pb["entry"]), "stop": float(pb["stop"]),
                "atr1m": float(atr1m), "rvol": float(br["rvol"]), "pct": float(br["pct"]),
                "score": float(score)
            })
            continue

        # Lane B: CG-only exception
        if src == "coingecko":
            if meta["vol_usd"] < VOL_USD_MIN_B:
                snapshot["reject_stats"]["B_vol"] += 1; continue
            pc1 = meta.get("pc_1h") or 0.0
            pc24 = meta.get("pc_24h") or 0.0
            if not (pc1 >= EXC_1H_MIN and pc24 >= EXC_24H_MIN):
                snapshot["reject_stats"]["B_momentum"] += 1; continue
            bars = get_1m_bars_any(tkr, meta, sym, limit=120)
            if not bars:
                snapshot["reject_stats"]["bars"] += 1; continue
            # Use last bar; synthetic RVOL already handled
            pb = micro_pullback_ok(bars)
            atr1m = atr_from_klines(bars, period=14)
            if not pb or atr1m <= 0.0:
                snapshot["reject_stats"]["B_momentum"] += 1; continue
            # score favors 1h speed and 24h persistence
            score = (1.0 + pc1*5.0) * (1.0 + pc24)
            exception_lane.append({
                "lane":"B","ticker":tkr,"symbol":sym or tkr,
                "entry": float(pb["entry"]), "stop": float(pb["stop"]),
                "atr1m": float(atr1m), "rvol": 3.0, "pct": 0.0,
                "score": float(score)
            })
            continue

        # Unknown or no meta
        if src == "none":
            snapshot["reject_stats"]["A_meta"] += 1
        else:
            snapshot["reject_stats"]["B_meta"] += 1

    # rank
    orderbook_lane.sort(key=lambda x: x["score"], reverse=True)
    exception_lane.sort(key=lambda x: x["score"], reverse=True)

    # choose up to 2 total, prefer lane A then B
    chosen: List[Dict[str, Any]] = []
    if orderbook_lane: chosen.append(orderbook_lane[0])
    if len(chosen) < 2 and len(orderbook_lane) > 1:
        chosen.append(orderbook_lane[1])
    if len(chosen) < 2 and exception_lane:
        chosen.append(exception_lane[0])

    # snapshot candidates (top 3 for visibility)
    snapshot["candidates"] = (orderbook_lane + exception_lane)[:3]
    snapshot["universe_count"] = len(orderbook_lane) + len(exception_lane)

    # outputs
    if not chosen:
        if blocked_451:
            snapshot["regime"] = {"ok": False, "reason": "binance-451-block"}
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        SIGNAL_PATH.write_text(json.dumps({"type":"C","text":"Hold and wait."}, indent=2))
        DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
        RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t0)*1000), "time": _now_iso()}, indent=2))
        print("Hold and wait. (No qualified candidates.)")
        return

    # size each pick with cap
    plan_lines = []
    total_cap = (STRONG_ALLOC if strong else WEAK_ALLOC) * equity
    remaining = min(total_cap, cash)
    for i, c in enumerate(chosen):
        entry = c["entry"]; stop = c["stop"]; atr = c["atr1m"]
        lane = c["lane"]
        # size
        usd = position_size(entry, stop, equity, remaining, strong, lane)
        remaining = max(0.0, remaining - usd)
        t1 = entry + 0.8 * atr
        t2 = entry + 1.5 * atr
        plan_lines.extend([
            f"• {c['ticker']} lane {lane}",
            f"  Entry {entry:.6f} | Stop {stop:.6f} | T1 {t1:.6f} | T2 {t2:.6f}",
            f"  Position ${usd:,.2f} | Trail after +1R"
        ])
        # update candidate with computed size (for summary consumers)
        c["t1"] = t1; c["t2"] = t2; c["usd"] = usd

        if remaining <= 0:
            break

    text = "\n".join(plan_lines) if plan_lines else "Hold and wait."
    SIGNAL_PATH.write_text(json.dumps({"type":"B","text": text}, indent=2))

    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
    DEBUG_PATH.write_text(json.dumps(DEBUG_LOG, indent=2))
    RUN_STATS.write_text(json.dumps({"elapsed_ms": int((perf_counter()-t0)*1000), "time": _now_iso()}, indent=2))
    print(text)

if __name__ == "__main__":
    main()