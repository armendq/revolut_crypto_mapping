# scripts/analyses.py
from __future__ import annotations

import json, math, os, time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- Config ----------------
VOL_USD_MIN = 3_000_000          # relaxed liquidity floor
MAX_UNI = 60                     # max coins to scan when expanding from CG
FLOOR_EQUITY = 40_000            # do not open new risk below this

# Aggressive 1m breakout (relaxed)
BREAKOUT_PCT_MIN = 0.010         # +1.0%
BREAKOUT_PCT_MAX = 0.060         # +6.0%
RVOL_1M_MIN = 2.5
MICRO_FADE_MAX = 0.010           # 1.0%

# Momentum path
HH_LOOKBACK_5M = 20              # bars
ROC_15M_MIN = 0.04               # +4%
RVOL_5M_MIN = 1.8

# Risk model
RISK_PCT = 0.012
CAP_OK = 0.60
CAP_WEAK = 0.30

OUT_DIR = "public_runs/latest"
os.makedirs(OUT_DIR, exist_ok=True)

CG = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "revolut-crypto-analyses/1.0 (+github actions)"}

# ---------------- HTTP session with retries ----------------
def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.headers.update(HEADERS)
    return s

HTTP = _session()

# ---------------- Utilities ----------------
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

# ---------------- Portfolio ----------------
def load_portfolio() -> Dict[str, float]:
    pf = read_json("data/portfolio.json")
    if pf and "equity" in pf and "cash" in pf:
        return {"equity": float(pf["equity"]), "cash": float(pf["cash"])}
    return {
        "equity": env_float("EQUITY", 41000.0),
        "cash": env_float("CASH", 32000.0),
    }

# ---------------- Mapping / Universe ----------------
def load_mapping_df() -> pd.DataFrame:
    # Try common locations; otherwise empty -> triggers CG expansion
    for p in ["data/mapping.csv", "data/mapping.json", "mapping/mapping.csv"]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p) if p.endswith(".csv") else pd.read_json(p)
                if "symbol" in df.columns and "cg_id" in df.columns:
                    df = df[["symbol", "cg_id"]].dropna()
                    df["symbol"] = df["symbol"].str.upper()
                    return df
            except Exception:
                pass
    return pd.DataFrame(columns=["symbol", "cg_id"])

def cg_top_markets(n: int = MAX_UNI) -> pd.DataFrame:
    # Expand universe from CG markets (market cap desc)
    per_page = min(250, n)
    r = HTTP.get(
        f"{CG}/coins/markets",
        params=dict(vs_currency="usd", order="market_cap_desc",
                    per_page=per_page, page=1, sparkline="false"),
        timeout=30,
    )
    r.raise_for_status()
    rows = []
    stables = {"usdt", "usdc", "dai", "tusd", "usde", "usdd", "fdusd", "susd", "lusd"}
    for it in r.json():
        sym = str(it.get("symbol", "")).upper()
        if sym.lower() in stables:
            continue
        vol = float(it.get("total_volume") or 0.0)
        if vol < VOL_USD_MIN:
            continue
        rows.append({
            "symbol": sym,
            "cg_id": it.get("id"),
            "price": float(it.get("current_price") or 0.0),
            "vol24": vol,
            "src": "coingecko",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.head(n).copy()
    return df

def cg_simple_price(ids: List[str]) -> Dict[str, Any]:
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i+100]
        r = HTTP.get(
            f"{CG}/simple/price",
            params=dict(
                ids=",".join(chunk),
                vs_currencies="usd",
                include_24hr_vol="true",
                include_last_updated_at="true",
            ),
            timeout=30,
        )
        r.raise_for_status()
        out.update(r.json() or {})
        time.sleep(0.2)
    return out

def build_universe(base_map: pd.DataFrame) -> pd.DataFrame:
    df = base_map.copy()
    # If mapping is tiny (<10), expand with CG top markets
    if len(df) < 10:
        extra = cg_top_markets(MAX_UNI)
        if not extra.empty:
            df = pd.concat([df, extra[["symbol", "cg_id"]]], ignore_index=True).drop_duplicates("symbol")
    # Enrich with live price/vol
    ids = df["cg_id"].dropna().tolist()
    prices = cg_simple_price(ids) if ids else {}
    rows = []
    for _, r in df.iterrows():
        cid = r["cg_id"]
        sym = r["symbol"].upper()
        p = prices.get(cid, {})
        rows.append({
            "symbol": sym,
            "cg_id": cid,
            "price": float(p.get("usd") or 0.0),
            "vol24": float(p.get("usd_24h_vol") or 0.0),
            "src": "coingecko",
        })
    uni = pd.DataFrame(rows)
    if uni.empty:
        return uni
    # Filter by volume (spread unknown on CG, allow)
    uni = uni[uni["vol24"] >= VOL_USD_MIN].reset_index(drop=True)
    # Always keep ETH & DOT for visibility
    force = uni[uni["symbol"].isin(["ETH", "DOT"])]
    uni = pd.concat([uni, force]).drop_duplicates("symbol").reset_index(drop=True)
    # Cap universe
    return uni.head(MAX_UNI).copy()

# ---------------- OHLC from CG ----------------
def cg_market_chart_1d(cg_id: str) -> pd.DataFrame:
    r = HTTP.get(f"{CG}/coins/{cg_id}/market_chart",
                 params=dict(vs_currency="usd", days=1), timeout=40)
    r.raise_for_status()
    js = r.json() or {}
    prices = js.get("prices") or []
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df["time"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("time").sort_index().drop(columns=["ts_ms"])
    # Build OHLC via resample of price
    ohlc1 = df.resample("1T").agg({"price": ["first", "max", "min", "last"]}).dropna()
    ohlc1.columns = ["open", "high", "low", "close"]
    return ohlc1

def ohlc_1m_5m(cg_id: str) -> Dict[str, pd.DataFrame]:
    df1 = cg_market_chart_1d(cg_id)
    if df1.empty:
        return {}
    df5 = df1.resample("5T").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return {"1m": df1.tail(360), "5m": df5.tail(200)}

# ---------------- Indicators ----------------
def atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < n + 1:
        return float("nan")
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])

def rvol(series: pd.Series, lookback: int = 15) -> float:
    if len(series) < lookback + 1:
        return float("nan")
    last = series.iloc[-1]
    avg = series.iloc[-(lookback+1):-1].mean()
    return (last / avg) if avg > 0 else float("nan")

def pct(a: float, b: float) -> float:
    if b == 0 or not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return (a - b) / b

# ---------------- Entries ----------------
def aggr_breakout_1m(df1: pd.DataFrame) -> Optional[Dict[str, float]]:
    if len(df1) < 20:
        return None
    c0, c1 = float(df1["close"].iloc[-1]), float(df1["close"].iloc[-2])
    move = pct(c0, c1)
    if not (BREAKOUT_PCT_MIN <= move <= BREAKOUT_PCT_MAX):
        return None
    low_last = float(df1["low"].iloc[-1])
    if (c0 - low_last) / c0 > MICRO_FADE_MAX:
        return None
    change_mag = df1["close"].diff().abs().fillna(0.0)
    rv = rvol(change_mag, 15)
    if not (rv >= RVOL_1M_MIN):
        return None
    hh = float(df1["high"].rolling(15).max().iloc[-2])
    if c0 <= hh:
        return None
    return {"entry": c0, "stop_hint": low_last, "atype": "aggr_1m"}

def momentum_5m(df5: pd.DataFrame, df1: pd.DataFrame) -> Optional[Dict[str, float]]:
    if len(df5) < HH_LOOKBACK_5M + 1 or len(df1) < 16:
        return None
    c0 = float(df5["close"].iloc[-1])
    hh = float(df5["high"].rolling(HH_LOOKBACK_5M).max().iloc[-2])
    if c0 <= hh:
        return None
    c1m = df1["close"]
    roc = pct(float(c1m.iloc[-1]), float(c1m.iloc[-15]))
    if roc < ROC_15M_MIN:
        return None
    rv = rvol(df5["close"].pct_change().abs().fillna(0.0), 12)
    if rv < RVOL_5M_MIN:
        return None
    low_last1m = float(df1["low"].iloc[-1])
    return {"entry": c0, "stop_hint": low_last1m, "atype": "mom_5m"}

# ---------------- Sizing ----------------
def position_size_usd(equity: float, cash: float, regime_ok: bool, entry: float, stop: float) -> float:
    cap = CAP_OK if regime_ok else CAP_WEAK
    max_alloc = (equity + cash) * cap
    risk = equity * RISK_PCT
    per_unit = max(entry - stop, entry * 0.005)
    qty = risk / per_unit
    usd = min(qty * entry, max_alloc)
    return max(0.0, float(usd))

# ---------------- Regime ----------------
def btc_regime() -> Dict[str, Any]:
    try:
        df = cg_market_chart_1d("bitcoin")
        if df.empty:
            return {"ok": False, "reason": "no-data", "last": None, "vwap": None, "ema9": None}
        close = df["close"]
        last = float(close.iloc[-1])
        ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
        vwap_proxy = float(close.ewm(span=30, adjust=False).mean().iloc[-1])
        ok = (last >= vwap_proxy) and (last >= ema9)
        return {"ok": ok, "reason": "" if ok else "btc-below-vwap-or-ema", "last": last, "vwap": vwap_proxy, "ema9": ema9}
    except Exception:
        return {"ok": False, "reason": "exception", "last": None, "vwap": None, "ema9": None}

# ---------------- Main ----------------
def main():
    t0 = time.time()
    portfolio = load_portfolio()
    base_map = load_mapping_df()
    uni = build_universe(base_map)

    dbg: List[Dict[str, Any]] = []
    cands: List[Dict[str, Any]] = []

    regime = btc_regime()

    for _, r in uni.iterrows():
        sym, cid = r["symbol"], r["cg_id"]
        if not cid:
            dbg.append({"symbol": sym, "kept": False, "note": "no-cg-id"})
            continue
        try:
            rows = ohlc_1m_5m(cid)
            if not rows:
                dbg.append({"symbol": sym, "kept": False, "note": "no-ohlc"})
                continue
            df1, df5 = rows["1m"], rows["5m"]
            last = float(df1["close"].iloc[-1])
            atr14 = atr(df1, 14)
            sig = aggr_breakout_1m(df1) or momentum_5m(df5, df1)

            entry = stop = None
            atype = None
            if sig:
                entry = float(sig["entry"])
                stop_hint = float(sig["stop_hint"])
                noise = max(entry * 0.006, (atr14 * 0.6) if math.isfinite(atr14) else entry * 0.006)
                stop = min(stop_hint, entry - noise)
                atype = sig["atype"]

            # DOT is staked (never propose a trade)
            note = "ok"
            if sym == "DOT":
                entry = stop = None
                atype = None
                note = "dot-staked"

            cands.append({
                "symbol": sym,
                "price": last,
                "atr": float(atr14) if math.isfinite(atr14) else None,
                "entry": entry,
                "stop": stop,
                "atype": atype,
                "vol24": float(r["vol24"] or 0.0),
            })
            dbg.append({"symbol": sym, "kept": True, "note": note})
            time.sleep(0.15)  # be gentle with CG
        except Exception as e:
            dbg.append({"symbol": sym, "kept": False, "note": f"err:{type(e).__name__}"})
            continue

    # Rank: prefer have-entry, then crude momentum (entry/price)
    def score(c):
        base = 10 if c["entry"] else 0
        mom = 0.0
        if c["entry"] and c["price"] > 0:
            mom = min(2.0, (c["entry"] / c["price"]) - 1.0)
        return base + mom

    cands_sorted = sorted(cands, key=score, reverse=True)
    chosen = next((c for c in cands_sorted if c["entry"]), None)

    signal = {"type": "C", "text": "Hold and wait."}
    if chosen:
        if portfolio["equity"] < FLOOR_EQUITY:
            signal = {"type": "C", "text": "Equity below floor; hold and wait."}
        else:
            entry = float(chosen["entry"])
            stop = float(chosen["stop"])
            atrv = float(chosen["atr"] or (entry * 0.01))
            t1 = entry + 0.8 * atrv
            t2 = entry + 1.5 * atrv
            size = position_size_usd(
                equity=portfolio["equity"],
                cash=portfolio["cash"],
                regime_ok=bool(regime["ok"]),
                entry=entry,
                stop=stop,
            )
            signal = {
                "type": "B",
                "text": "Breakout entry.",
                "symbol": chosen["symbol"],
                "entry": round(entry, 6),
                "stop": round(stop, 6),
                "t1": round(t1, 6),
                "t2": round(t2, 6),
                "size_usd": round(size, 2),
                "trail_after_R": 1.0,
                "atype": chosen.get("atype"),
            }

    summary = {
        "generated_at_utc": utcnow(),
        "regime": regime,
        "equity": float(portfolio["equity"]),
        "cash": float(portfolio["cash"]),
        "candidates": cands_sorted[:10],
        "signals": signal,
        "run_stats": {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "time": utcnow(),
        },
    }

    write_json(os.path.join(OUT_DIR, "summary.json"), summary)
    write_json(os.path.join(OUT_DIR, "signals.json"), {"signals": signal})
    write_json(os.path.join(OUT_DIR, "run_stats.json"), summary["run_stats"])
    write_json(os.path.join(OUT_DIR, "debug_scan.json"), dbg)
    write_json(os.path.join(OUT_DIR, "market_snapshot.json"), {
        "universe_size": int(len(uni)),
        "expanded_from_mapping": int(max(0, len(uni) - len(base_map))),
    })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Emit a minimal summary so downstream steps still have a file
        err = {
            "generated_at_utc": utcnow(),
            "regime": {"ok": False, "reason": "exception", "last": None, "vwap": None, "ema9": None},
            "equity": env_float("EQUITY", 41000.0),
            "cash": env_float("CASH", 32000.0),
            "candidates": [],
            "signals": {"type": "C", "text": f"Error: {type(e).__name__}"},
            "run_stats": {"elapsed_ms": 0, "time": utcnow()},
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        write_json(os.path.join(OUT_DIR, "summary.json"), err)
        write_json(os.path.join(OUT_DIR, "signals.json"), {"signals": err["signals"]})
        write_json(os.path.join(OUT_DIR, "run_stats.json"), err["run_stats"])
        write_json(os.path.join(OUT_DIR, "debug_scan.json"), [{"note": f"fatal:{type(e).__name__}"}])
        # Do not re-raise—keep workflow green while still publishing artifacts