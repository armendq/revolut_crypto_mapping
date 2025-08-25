#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust market scanner for Revolut-listed coins.

- Reads mapping (ticker -> {binance_symbol, coingecko_id})
- Tries Binance klines FIRST when binance_symbol exists (ticker is always in URL)
- Falls back to CoinGecko (market_chart) if:
    * there is no binance_symbol
    * Binance returns HTTP 4xx/5xx or rate limit
- Computes:
    * 24h change, 6–12h ramp %, 1h ramp %, ATR (from closes), breakout vs recent range,
      volume (from Binance where possible), and a composite momentum score
- Produces:
    * public_runs/latest/summary.json   (status, regime, signals, candidates)
    * public_runs/latest/signals.json   (raw picks and reasons)
    * public_runs/latest/market_snapshot.json (quick snapshot)
    * public_runs/latest/run_stats.json (timing + api stats)
    * public_runs/latest/debug_scan.json (per-ticker debug)

Notes
- Keep requests light: binance 15m klines limit=288 (~3 days), coingecko 2 days hourly.
- Be resilient: retry with exponential backoff for transient errors.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

import requests

# --------------------------- Config ---------------------------------

DEFAULT_MAPPING_FILES = [
    "mapping/generated_mapping.json",
    "data/mapping.json",
    "mapping.json",
]
OUT_DIR = "public_runs/latest"

USER_AGENT = "revolut-crypto-analyses/1.0 (+github-actions)"
BINANCE_BASE = "https://api.binance.com"
BINANCE_VISION = "https://data-api.binance.vision"
BINANCE_INTERVAL = "15m"
BINANCE_LIMIT = 288  # 3 days of 15m candles

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

REQUEST_TIMEOUT = 15
RETRY_MAX = 4
RETRY_BASE_SLEEP = 1.2


# --------------------------- Helpers --------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_mapping() -> Dict[str, Dict[str, str]]:
    for p in DEFAULT_MAPPING_FILES:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Accept either {ticker: {...}} or {"assets":[{"ticker":...}]}
            if isinstance(data, dict) and "assets" in data:
                mapping = {}
                for a in data["assets"]:
                    t = a.get("ticker") or a.get("symbol") or a.get("code")
                    if not t:
                        continue
                    mapping[t.upper()] = {
                        "binance_symbol": (a.get("binance_symbol") or a.get("binanceSymbol")),
                        "coingecko_id": (a.get("coingecko_id") or a.get("coingeckoId")),
                        "name": a.get("name") or t.upper(),
                    }
                return mapping
            elif isinstance(data, dict):
                # assume already {ticker:{...}}
                return {k.upper(): v for k, v in data.items()}
    print("[analyses] WARNING: no mapping file found; scanning nothing.", file=sys.stderr)
    return {}


def _retry_get(url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None
               ) -> requests.Response:
    headers = dict(headers or {})
    headers.setdefault("User-Agent", USER_AGENT)
    last_exc = None
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            # 451/429/5xx -> backoff and retry
            code = r.status_code
            print(f"[analyses] {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC GET fail: {r.url} http {code} (try {attempt})")
            if code in (429, 451) or (500 <= code < 600) or code in (400, 403):
                time.sleep(RETRY_BASE_SLEEP * attempt)
                continue
            # hard failure for other 4xx
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            print(f"[analyses] error GET {url}: {e} (try {attempt})")
            time.sleep(RETRY_BASE_SLEEP * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"GET failed after retries: {url}")


# --------------------------- Data models -----------------------------

@dataclass
class Candle:
    ts: int  # ms
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


def parse_binance_klines(raw: List[List[Any]]) -> List[Candle]:
    out: List[Candle] = []
    for row in raw:
        # https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        ts_ms = int(row[0])
        o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4])
        v = float(row[5])
        out.append(Candle(ts=ts_ms, open=o, high=h, low=l, close=c, volume=v))
    return out


def parse_coingecko_market_chart(raw: Dict[str, Any]) -> List[Candle]:
    # coingecko returns arrays of [timestamp, value]
    prices = raw.get("prices") or []
    out: List[Candle] = []
    for i, (ts_ms, price) in enumerate(prices):
        # we do not have OHLC/volume for hourly points; approximate with close-only candles
        c = float(price)
        # synth open/hl from neighbors (best effort)
        prev = float(prices[i-1][1]) if i > 0 else c
        o = prev
        h = max(o, c)
        l = min(o, c)
        out.append(Candle(ts=int(ts_ms), open=o, high=h, low=l, close=c, volume=None))
    return out


# --------------------------- Sources --------------------------------

def fetch_binance_klines(binance_symbol: str, interval: str = BINANCE_INTERVAL, limit: int = BINANCE_LIMIT
                         ) -> List[Candle]:
    """
    Always provides `symbol` in URL, as requested.
    Tries Binance main first; if it rate-limits, tries binance.vision mirror.
    """
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
    url_main = f"{BINANCE_BASE}/api/v3/klines"
    url_mirror = f"{BINANCE_VISION}/api/v3/klines"

    try:
        r = _retry_get(url_main, params=params)
        return parse_binance_klines(r.json())
    except Exception:
        # fallback to mirror
        r = _retry_get(url_mirror, params=params)
        return parse_binance_klines(r.json())


def fetch_coingecko_ohlc(cg_id: str) -> List[Candle]:
    # 2 days hourly gives enough to compute 24h & 6–12h ramps
    url = f"{COINGECKO_BASE}/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": "2", "interval": "hourly"}
    r = _retry_get(url, params=params)
    return parse_coingecko_market_chart(r.json())


# --------------------------- Indicators ------------------------------

def atr_from_closes(closes: List[float], window: int = 14) -> float:
    if len(closes) < window + 1:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        trs.append(abs(cur - prev))
    if len(trs) < window:
        return 0.0
    return statistics.fmean(trs[-window:])


def pct_change(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a / b - 1.0) * 100.0


def rolling_max(xs: List[float], window: int) -> float:
    if len(xs) < window:
        return max(xs) if xs else 0.0
    return max(xs[-window:])


def rolling_min(xs: List[float], window: int) -> float:
    if len(xs) < window:
        return min(xs) if xs else 0.0
    return min(xs[-window:])


# --------------------------- Scoring ---------------------------------

@dataclass
class ScanResult:
    ticker: str
    name: str
    source: str  # "binance" or "coingecko"
    last: float
    change_24h: float
    ramp_12h: float
    ramp_6h: float
    ramp_1h: float
    atr: float
    breakout: float  # distance vs 3d high (%)
    stop_hint: float
    entry_hint: float
    score: float
    reason: str


def score_asset(candles: List[Candle]) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Returns: last, ch24, r12, r6, r1, atr, breakout_pct, stop_hint
    """
    closes = [c.close for c in candles]
    if len(closes) < 10:
        last = closes[-1] if closes else 0.0
        return last, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, last * 0.9

    last = closes[-1]
    prev_24h = closes[-96] if len(closes) >= 96 else closes[0]  # 15m*96=24h
    prev_12h = closes[-48] if len(closes) >= 48 else closes[0]
    prev_6h = closes[-24] if len(closes) >= 24 else closes[0]
    prev_1h = closes[-4] if len(closes) >= 4 else closes[0]

    ch24 = pct_change(last, prev_24h)
    r12 = pct_change(last, prev_12h)
    r6 = pct_change(last, prev_6h)
    r1 = pct_change(last, prev_1h)

    atrv = atr_from_closes(closes, 14)
    # breakout vs recent (3 days -> 288 bars)
    lookback = min(len(closes), 288)
    hh = max(closes[-lookback:])
    breakout_pct = pct_change(last, hh)

    # stop: use recent swing low over last 12 bars as proxy (breakout bar low once we detect B)
    stop_hint = rolling_min(closes, 12)

    return last, ch24, r12, r6, r1, atrv, breakout_pct, stop_hint


def composite_score(ch24: float, r12: float, r6: float, r1: float, breakout: float, atrv: float, last: float) -> float:
    # Heuristic scoring: prefer steady multi-hour ramps (12h + 6h) with positive 1h;
    # penalize being far above recent HH (to avoid chasing exhaustion);
    # normalize ATR relative to price.
    if last <= 0:
        return -1e9
    atr_rel = (atrv / last) * 100.0 if last > 0 else 0.0
    score = (
        0.45 * r12 +
        0.25 * r6 +
        0.15 * r1 +
        0.10 * ch24 -
        0.20 * max(0.0, breakout) +  # if already above HH, be cautious
        0.15 * min(atr_rel, 5.0)     # we want some range to set stops/targets
    )
    return score


# --------------------------- Scan pipeline ---------------------------

def scan_universe(mapping: Dict[str, Dict[str, str]]) -> Tuple[List[ScanResult], Dict[str, Any]]:
    results: List[ScanResult] = []
    debug: Dict[str, Any] = {}
    api_stats = {"binance_ok": 0, "binance_fail": 0, "coingecko_ok": 0, "coingecko_fail": 0}

    tickers = sorted(mapping.keys())
    print(f"[analyses] universe: {len(tickers)}")

    for idx, t in enumerate(tickers, 1):
        meta = mapping.get(t, {})
        bsym = (meta.get("binance_symbol") or "").upper().strip()
        cg_id = (meta.get("coingecko_id") or "").strip()
        name = meta.get("name") or t

        candles: List[Candle] = []
        source_used = ""

        # Prefer Binance if we have a symbol
        if bsym:
            try:
                # IMPORTANT: ticker/binance_symbol is always included in URL params
                candles = fetch_binance_klines(bsym)
                source_used = "binance"
                api_stats["binance_ok"] += 1
            except Exception as e:
                api_stats["binance_fail"] += 1
                debug.setdefault(t, {})["binance_error"] = str(e)

        # Fallback to CoinGecko (or primary if no Binance mapping)
        if not candles and cg_id:
            try:
                candles = fetch_coingecko_ohlc(cg_id)
                source_used = "coingecko"
                api_stats["coingecko_ok"] += 1
            except Exception as e:
                api_stats["coingecko_fail"] += 1
                debug.setdefault(t, {})["coingecko_error"] = str(e)

        if not candles:
            debug.setdefault(t, {})["skipped"] = "no data from any source"
            continue

        last, ch24, r12, r6, r1, atrv, breakout, stop_hint = score_asset(candles)
        entry_hint = last  # conservative: consider current for candidate display
        score = composite_score(ch24, r12, r6, r1, breakout, atrv, last)

        reason = (
            f"r12={r12:.1f}%, r6={r6:.1f}%, r1={r1:.1f}%, 24h={ch24:.1f}%, "
            f"ATR={atrv:.4f}, breakout={breakout:.2f}%"
        )

        results.append(ScanResult(
            ticker=t, name=name, source=source_used, last=last, change_24h=ch24,
            ramp_12h=r12, ramp_6h=r6, ramp_1h=r1, atr=atrv, breakout=breakout,
            stop_hint=stop_hint, entry_hint=entry_hint, score=score, reason=reason
        ))

        if idx % 25 == 0:
            print(f"[analyses] scanned {idx}/{len(tickers)}")

    # sort best first
    results.sort(key=lambda r: r.score, reverse=True)
    debug["api_stats"] = api_stats
    return results, debug


# --------------------------- Regime & Signals ------------------------

def compute_regime(results: List[ScanResult]) -> Dict[str, Any]:
    # Simple regime: median 24h change across assets; BTC/ETH presence if available
    if not results:
        return {"ok": False, "reason": "no data", "breadth": 0.0}

    ch24s = [r.change_24h for r in results if not math.isnan(r.change_24h)]
    breadth = sum(1 for x in ch24s if x > 0) / len(ch24s) if ch24s else 0.0
    ok = breadth >= 0.45  # bullish-enough breadth
    return {"ok": ok, "breadth": round(breadth, 3)}


def build_signals(results: List[ScanResult], regime: Dict[str, Any]) -> Dict[str, Any]:
    # “Breakout signal” if we find something strong with moderate breakout premium
    picks: List[Dict[str, Any]] = []
    for r in results[:25]:
        # prefer multi-hour ramps, not blow-off: keep breakout <= +1% (at/near HH)
        if r.ramp_12h > 12 and r.ramp_6h > 6 and r.ramp_1h > -2 and r.breakout <= 1.2:
            # make an actionable suggestion skeleton
            atr = r.atr or max(0.001, r.last * 0.01)
            t1 = r.last + 0.8 * atr
            t2 = r.last + 1.5 * atr
            picks.append({
                "ticker": r.ticker,
                "entry": round(r.entry_hint, 8),
                "stop": round(max(0.00000001, r.stop_hint), 8),
                "t1": round(t1, 8),
                "t2": round(t2, 8),
                "atr": round(atr, 8),
                "reason": r.reason,
                "source": r.source,
            })

    signals: Dict[str, Any] = {}
    if picks and regime.get("ok", False):
        signals["type"] = "B"
        signals["picks"] = picks[:3]
    else:
        signals["type"] = "C"
        # top candidates even if regime weak
        cands = []
        for r in results[:15]:
            cands.append({
                "ticker": r.ticker,
                "entry": round(r.entry_hint, 8),
                "stop": round(max(0.00000001, r.stop_hint), 8),
                "atr": round(r.atr, 8),
                "r12": round(r.ramp_12h, 2),
                "r6": round(r.ramp_6h, 2),
                "r1": round(r.ramp_1h, 2),
                "score": round(r.score, 3),
                "src": r.source,
            })
        signals["candidates"] = cands
    return signals


# --------------------------- Main -----------------------------------

def main() -> int:
    t0 = time.time()
    ensure_out_dir(OUT_DIR)

    mapping = load_mapping()
    if not mapping:
        # still produce minimal files to keep pipeline green
        minimal = {"status": "ok", "generated_at": now_utc_iso(), "note": "empty mapping"}
        with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(minimal, f, indent=2)
        with open(os.path.join(OUT_DIR, "run_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"duration_sec": round(time.time() - t0, 2)}, f, indent=2)
        return 0

    results, debug = scan_universe(mapping)
    regime = compute_regime(results)
    signals = build_signals(results, regime)

    # market snapshot
    snapshot = []
    for r in results[:30]:
        snapshot.append({
            "ticker": r.ticker,
            "last": round(r.last, 8),
            "24h": round(r.change_24h, 2),
            "r12": round(r.ramp_12h, 2),
            "r6": round(r.ramp_6h, 2),
            "r1": round(r.ramp_1h, 2),
            "atr": round(r.atr, 8),
            "score": round(r.score, 3),
            "src": r.source,
        })

    summary = {
        "status": "ok",
        "generated_at": now_utc_iso(),
        "regime": regime,
        "signals": signals,
        "meta": {
            "universe": len(mapping),
            "scanned": len(results),
            "binance_interval": BINANCE_INTERVAL,
            "binance_limit": BINANCE_LIMIT,
        },
    }

    # Write outputs
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(OUT_DIR, "signals.json"), "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2)

    with open(os.path.join(OUT_DIR, "market_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    with open(os.path.join(OUT_DIR, "debug_scan.json"), "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)

    with open(os.path.join(OUT_DIR, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "duration_sec": round(time.time() - t0, 2),
            "generated_at": now_utc_iso(),
            **(debug.get("api_stats") or {})
        }, f, indent=2)

    print(f"[analyses] wrote summary with {len(signals.get('candidates', []))} candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())