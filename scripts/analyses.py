#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses runner (robust momentum scan)
- Accepts multiple mapping shapes from data/revolut_mapping.json
- Scans Binance first (with symbol=...), falls back to Coinbase
- Uses 15m candles to detect ~8h momentum moves and ATR for stops/targets
- Filters by liquidity/spread; excludes ETH and DOT from rotation (DOT staked)
- Outputs:
    public_runs/latest/summary.json
    artifacts/{market_snapshot.json, run_stats.json, debug_scan.json}
    data/signals.json
"""

from __future__ import annotations
import os, json, time, math, statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
PUB = ROOT / "public_runs" / "latest"
for p in (DATA, ART, PUB):
    p.mkdir(parents=True, exist_ok=True)

SNAP_PATH = ART / "market_snapshot.json"
SIGNAL_PATH = DATA / "signals.json"
RUN_STATS = ART / "run_stats.json"
DEBUG_PATH = ART / "debug_scan.json"

# -------------------- Config --------------------
USER_AGENT = "rev-analyses/2.0 (+github.com/armendq/revolut_crypto_mapping)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
TIMEOUT = 12
MAX_RETRIES = 4
SLEEP_BETWEEN = 0.18  # conservative rate limiting

# Binance endpoints (try multiple to dodge regional issues)
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api4.binance.com",
    "https://api-gcp.binance.com",
    "https://data-api.binance.vision",
]
COINBASE_BASE = "https://api.exchange.coinbase.com"

# Scan logic
INTERVAL = "15m"        # 15-minute bars
LIMIT = 96              # 24h of 15m bars; use last 32 bars for ~8h
WINDOW_8H = 32

# Candidate filters (tuned to catch 8h movers like DIMO/ACS spikes)
MIN_MOVE_8H = 12.0         # % move over ~8h (lowered to not miss)
MIN_USD_VOL_8H = 300_000   # estimated from last 32 bars
MAX_SPREAD_PCT = 1.0       # skip if proxy spread > 1%
EXCLUDE_TICKERS = {"ETH", "DOT"}  # ETH not rotated; DOT staked/untouchable

# Sizing and guards
RISK_PCT = 0.012
STRONG_ALLOC = 0.60
WEAK_ALLOC = 0.30
EQUITY_FLOOR = 40000.0  # per your instruction

DEBUG_LOG: List[Dict[str, Any]] = []

# -------------------- Helpers --------------------
def log(stage: str, **kw):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage}
    rec.update(kw)
    DEBUG_LOG.append(rec)

def write_json(path: Path, obj: Any):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def pct(a: float, b: float) -> float:
    if not a or math.isnan(a) or math.isnan(b):
        return 0.0
    return (b - a) / a * 100.0

def median(vals: List[float]) -> float:
    return statistics.median(vals) if vals else 0.0

# -------------------- HTTP --------------------
def http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=TIMEOUT)
            code = r.status_code
            if code == 200:
                log("http", url=r.url, code=code, try_no=attempt)
                return r.json()
            log("http_err", url=r.url, code=code, try_no=attempt, text=r.text[:200])
        except Exception as e:
            log("http_exc", url=url, err=str(e), try_no=attempt)
        time.sleep(min(2 ** (attempt - 1) * 0.5, 3.0))
    return None

# -------------------- Market sources --------------------
def binance_klines(symbol: str, interval: str = INTERVAL, limit: int = LIMIT) -> List[List[Any]]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/klines"
        js = http_get_json(url, params=params)
        time.sleep(SLEEP_BETWEEN)
        if isinstance(js, list) and js:
            return js
    return []

def binance_24h(symbol: str) -> Optional[Dict[str, Any]]:
    params = {"symbol": symbol}
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/ticker/24hr"
        js = http_get_json(url, params=params)
        time.sleep(SLEEP_BETWEEN)
        if isinstance(js, dict) and js.get("lastPrice") is not None:
            try:
                last = float(js["lastPrice"])
                vol_base = float(js.get("volume", 0.0))
                return {"last": last, "vol_usd": last * vol_base}
            except Exception:
                continue
    return None

def binance_book(symbol: str) -> Optional[Tuple[float, float]]:
    params = {"symbol": symbol, "limit": 5}
    for base in BINANCE_BASES:
        url = f"{base}/api/v3/depth"
        js = http_get_json(url, params=params)
        time.sleep(SLEEP_BETWEEN)
        if isinstance(js, dict) and "bids" in js and "asks" in js and js["bids"] and js["asks"]:
            try:
                bid = float(js["bids"][0][0])
                ask = float(js["asks"][0][0])
                return bid, ask
            except Exception:
                continue
    return None

def cb_pid(ticker: str) -> str:
    return f"{ticker.upper()}-USD"

def coinbase_candles(ticker: str, granularity: int = 900, limit: int = LIMIT) -> List[List[Any]]:
    # Coinbase returns [ time, low, high, open, close, volume ] newest first sometimes
    url = f"{COINBASE_BASE}/products/{cb_pid(ticker)}/candles?granularity={granularity}"
    js = http_get_json(url)
    time.sleep(SLEEP_BETWEEN)
    if isinstance(js, list) and js:
        try:
            rows = sorted(js, key=lambda r: r[0])  # ascending
            return rows[-limit:]
        except Exception:
            return []
    return []

def coinbase_stats(ticker: str) -> Optional[Dict[str, float]]:
    url = f"{COINBASE_BASE}/products/{cb_pid(ticker)}/stats"
    js = http_get_json(url)
    time.sleep(SLEEP_BETWEEN)
    if isinstance(js, dict) and "last" in js:
        try:
            last = float(js["last"])
            vol_base = float(js.get("volume", 0.0))
            return {"last": last, "vol_usd": last * vol_base}
        except Exception:
            return None
    return None

def coinbase_book(ticker: str) -> Optional[Tuple[float, float]]:
    url = f"{COINBASE_BASE}/products/{cb_pid(ticker)}/book?level=1"
    js = http_get_json(url)
    time.sleep(SLEEP_BETWEEN)
    if isinstance(js, dict) and js.get("bids") and js.get("asks"):
        try:
            bid = float(js["bids"][0][0]); ask = float(js["asks"][0][0])
            return bid, ask
        except Exception:
            return None
    return None

# -------------------- TA --------------------
def atr_from_klines_15m(rows: List[List[Any]], n: int = 14) -> float:
    # rows format:
    # Binance: [openTime, open, high, low, close, volume, closeTime, ...]
    # Coinbase: [time, low, high, open, close, volume]
    if len(rows) < n + 2:
        return float("nan")
    trs: List[float] = []
    # Normalize accessors
    def hlc(i) -> Tuple[float, float, float]:
        r = rows[i]
        if len(r) >= 6 and isinstance(r[0], (int, float)) and isinstance(r[1], (int, float)):
            # detect source by field types; both are numeric
            # assume indices:
            # Binance: [0,1,2,3,4,5,...]; Coinbase: [0,1,2,3,4,5]
            # We'll map by length position:
            if len(r) >= 12:  # Binance style
                h, l, c = as_float(r[2]), as_float(r[3]), as_float(r[4])
            else:             # Coinbase style
                l, h, o, c = as_float(r[1]), as_float(r[2]), as_float(r[3]), as_float(r[4])
            return h, l, c
        return float("nan"), float("nan"), float("nan")

    # seed prev close
    _, _, prev_close = hlc(0)
    for i in range(1, len(rows)):
        h, l, c = hlc(i)
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < n:
        return float("nan")
    window = trs[-n:]
    return sum(window) / len(window)

def last_candle_hlc(rows: List[List[Any]]) -> Tuple[float, float, float]:
    r = rows[-1]
    if len(r) >= 12:  # Binance style
        return as_float(r[2]), as_float(r[3]), as_float(r[4])
    else:  # Coinbase style
        return as_float(r[2]), as_float(r[1]), as_float(r[4])  # h, l, c

def dollar_vol_from_rows(rows: List[List[Any]]) -> float:
    s = 0.0
    if not rows:
        return s
    if len(rows[-1]) >= 12:  # Binance
        for r in rows:
            c = as_float(r[4]); v = as_float(r[5]); s += c * v
    else:  # Coinbase (volume is base)
        for r in rows:
            c = as_float(r[4]); v = as_float(r[5]); s += c * v
    return s

def spread_proxy_pct_from_last(rows: List[List[Any]]) -> float:
    h, l, _ = last_candle_hlc(rows)
    if l <= 0:
        return 999.0
    return (h / l - 1.0) * 100.0

# -------------------- Mapping --------------------
def load_revolut_mapping() -> List[Dict[str, str]]:
    """
    Accept:
      - list of dicts with keys: ticker, binance (or binance_symbol)
      - list of strings: ["BTC", "SOL", ...]
      - dict of { "BTC": "Bitcoin", ... }
    Build symbol default as {ticker}USDT.
    """
    path = DATA / "revolut_mapping.json"
    if not path.exists():
        raise FileNotFoundError("data/revolut_mapping.json missing")
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, str]] = []

    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                t = (x.get("ticker") or x.get("revolut_ticker") or "").upper()
                if not t:
                    continue
                b = (x.get("binance") or x.get("binance_symbol") or f"{t}USDT").upper()
                out.append({"ticker": t, "binance": b})
            elif isinstance(x, str):
                t = x.upper()
                out.append({"ticker": t, "binance": f"{t}USDT"})
    elif isinstance(obj, dict):
        for t in obj.keys():
            tt = t.upper()
            out.append({"ticker": tt, "binance": f"{tt}USDT"})
    else:
        raise ValueError("Unsupported mapping shape")

    # de-dup
    seen = set()
    uniq = []
    for m in out:
        key = (m["ticker"], m["binance"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq

# -------------------- Regime --------------------
def regime_ok() -> bool:
    try:
        rows = binance_klines("BTCUSDT", interval="1h", limit=200)
        if not rows:
            rows = coinbase_candles("BTC", granularity=3600, limit=200)
        closes = []
        for r in rows:
            if len(r) >= 12:  # Binance
                closes.append(as_float(r[4]))
            else:
                closes.append(as_float(r[4]))
        if len(closes) < 50:
            return True
        # EMA50
        ema = closes[0]
        alpha = 2 / (50 + 1)
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return closes[-1] >= ema
    except Exception as e:
        log("regime_err", err=str(e))
        return True

# -------------------- Evaluation --------------------
def get_rows_15m(ticker: str, binance_sym: str) -> List[List[Any]]:
    rows = binance_klines(binance_sym, interval=INTERVAL, limit=LIMIT)
    if not rows:
        rows = coinbase_candles(ticker, granularity=900, limit=LIMIT)
    return rows

def evaluate(ticker: str, binance_sym: str) -> Optional[Dict[str, Any]]:
    # Skip excluded
    if ticker in EXCLUDE_TICKERS:
        return None

    rows = get_rows_15m(ticker, binance_sym)
    if not rows or len(rows) < WINDOW_8H + 2:
        return None

    # 8h window
    sub = rows[-WINDOW_8H:]
    start_close = as_float(sub[0][4])
    end_close = as_float(sub[-1][4])
    move8h = pct(start_close, end_close)

    atr = atr_from_klines_15m(rows, n=14)
    if math.isnan(atr) or atr <= 0:
        return None

    usd_vol_8h = dollar_vol_from_rows(sub)
    spread_px = spread_proxy_pct_from_last(rows)

    # hard filters
    if move8h < MIN_MOVE_8H:
        return None
    if usd_vol_8h < MIN_USD_VOL_8H:
        return None
    if spread_px > MAX_SPREAD_PCT:
        return None

    # entry = last close; stop = last bar low (proxy for breakout bar low)
    h, l, c = last_candle_hlc(rows)
    entry = c
    stop = max(1e-12, l)

    # Additional info
    # Try orderbook spread sanity (not required)
    bid_ask = binance_book(binance_sym) or coinbase_book(ticker)
    if bid_ask:
        bid, ask = bid_ask
        if bid > 0 and ask > 0:
            book_spread = (ask - bid) / ((ask + bid) / 2) * 100.0
        else:
            book_spread = None
    else:
        book_spread = None

    return {
        "ticker": ticker,
        "symbol": binance_sym,
        "entry": round(entry, 10),
        "stop": round(stop, 10),
        "atr": round(atr, 10),
        "T1": round(entry + 0.8 * atr, 10),
        "T2": round(entry + 1.5 * atr, 10),
        "move8h_pct": round(move8h, 3),
        "usd_vol_8h": round(usd_vol_8h, 2),
        "spread_proxy_pct": round(spread_px, 3),
        "book_spread_pct": None if book_spread is None else round(book_spread, 3),
        "src": "binance_or_cb_15m"
    }

def position_size(entry: float, stop: float, equity: float, cash: float, strong: bool) -> float:
    risk_dollars = equity * RISK_PCT
    dist = max(entry - stop, 1e-10)
    qty = risk_dollars / dist
    usd = qty * entry
    cap = STRONG_ALLOC if strong else WEAK_ALLOC
    return max(0.0, min(usd, equity * cap, cash))

# -------------------- Main --------------------
def main():
    t0 = time.time()
    EQUITY = as_float(os.getenv("EQUITY", "41000") or 41000.0)
    CASH = as_float(os.getenv("CASH", "32000") or 32000.0)

    # Floor guard
    snapshot = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "equity": EQUITY,
        "cash": CASH,
        "equity_floor": EQUITY_FLOOR,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "candidates": []
    }

    if EQUITY <= EQUITY_FLOOR or (EQUITY - EQUITY_FLOOR) < 1000.0:
        snapshot["breach"] = True
        snapshot["breach_reason"] = "equity<=floor or buffer<1k"
        write_json(SNAP_PATH, snapshot)
        write_json(SIGNAL_PATH, {"type": "A", "text": "Raise cash now. Halt new entries; trim weakest on breaks."})
        write_json(RUN_STATS, {"elapsed_ms": int((time.time() - t0) * 1000), "time": snapshot["time_utc"]})
        write_json(DEBUG_PATH, DEBUG_LOG)
        # still publish a minimal public summary
        write_json(PUB / "summary.json", {
            "status": "ok",
            "generated_at_utc": snapshot["time_utc"],
            "regime": {"ok": False, "reason": "floor_guard"},
            "equity": EQUITY, "cash": CASH,
            "signals": {"type": "C", "text": "Hold and wait."},
            "candidates": []
        })
        print("Raise cash now.")
        return

    # Regime
    strong = regime_ok()
    snapshot["regime"] = {"ok": strong}

    # Mapping
    try:
        mapping = load_revolut_mapping()
    except Exception as e:
        log("mapping_err", err=str(e))
        # Fail safe
        write_json(SIGNAL_PATH, {"type": "C", "text": "Hold