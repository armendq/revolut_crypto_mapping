#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight market scan for Revolut-available coins.
- Pure stdlib (urllib, json, time, datetime, pathlib, math)
- Pulls basic market data from CoinGecko's public markets endpoint
- Produces:
    public_runs/latest/market_snapshot.json
    public_runs/latest/debug_scan.json
    public_runs/latest/run_stats.json
    public_runs/latest/signals.json
    public_runs/latest/summary.json

Signals:
- If any coin meets a strong breakout condition (near 24h high with big 24h range
  and elevated volume proxy), emits signals.type == "B" with a single best pick.
- Otherwise emits signals.type == "C" with ranked candidates.

“ATR” proxy:
- Uses 24h True Range approximation: atr_24h = high_24h - low_24h
- T1 = entry + 0.8 * atr_24h, T2 = entry + 1.5 * atr_24h

Robustness:
- Unknown/failed coins are skipped.
- Any exception is caught and folded into debug + a minimal summary so the
  workflow never fails due to data hiccups.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Configuration
# -------------------------

OUT_DIR = Path("public_runs/latest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VS_CCY = "usd"

# Universe mapping: {symbol: coingecko_id}
# NOTE: If you already maintain a mapping file in the repo, you can load it here
# to override/extend this default dictionary.
DEFAULT_UNIVERSE: Dict[str, str] = {
    # large caps / majors
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "DOT": "polkadot",
    "IMX": "immutable-x",
    "ARB": "arbitrum",
    "OP": "optimism",
    # smaller / requested
    "ACS": "access-protocol",
    "NEON": "neon-evm",           # best-effort id
    "SHDW": "genesysgo-shadow",   # best-effort id
    # add more if needed
}

# Breakout / candidate heuristics
NEAR_HIGH_PCT = 1.0       # price within 1% of 24h high -> “near high”
MIN_RANGE_PCT = 4.0       # (high-low)/price >= 4% -> meaningful range
MIN_PCT_24H = 5.0         # 24h pct change at least this for candidates

# A tiny throttle for polite API usage
REQUEST_SLEEP_SEC = 0.2


# -------------------------
# Helpers
# -------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_get_json(url: str, timeout: float = 20.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "revolut-crypto-mapping-analyses/1.0 (+https://github.com/armendq)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(url, resp.status, "Non-200", resp.headers, None)
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class CoinRow:
    symbol: str
    id: str
    price: Optional[float]
    high24: Optional[float]
    low24: Optional[float]
    pct24: Optional[float]
    mcap: Optional[float]
    vol24: Optional[float]


def fetch_markets(cg_ids: List[str]) -> Dict[str, CoinRow]:
    """
    Query CoinGecko markets for a batch of ids.
    Returns mapping by id -> CoinRow
    """
    out: Dict[str, CoinRow] = {}
    if not cg_ids:
        return out

    # CoinGecko markets endpoint (ids must be comma-separated list of ids)
    ids_param = ",".join(sorted(set(cg_ids)))
    url = (
        "https://api.coingecko.com/api/v3/coins/markets?"
        + urllib.parse.urlencode(
            {
                "vs_currency": VS_CCY,
                "ids": ids_param,
                "price_change_percentage": "24h",
                "per_page": len(cg_ids),
                "sparkline": "false",
            }
        )
    )

    data = []
    try:
        data = http_get_json(url)
    except Exception as e:
        # If it fails entirely, return empty; caller will handle
        OUT_DIR.joinpath("debug_scan.json").write_text(
            json.dumps({"error": f"markets fetch failed: {type(e).__name__}: {e}"}, indent=2)
        )
        return out

    for row in data:
        # row keys: id, symbol, current_price, high_24h, low_24h, price_change_percentage_24h_in_currency, market_cap, total_volume, ...
        cid = str(row.get("id") or "")
        sym = str(row.get("symbol") or "").upper()
        out[cid] = CoinRow(
            symbol=sym,
            id=cid,
            price=safe_float(row.get("current_price")),
            high24=safe_float(row.get("high_24h")),
            low24=safe_float(row.get("low_24h")),
            pct24=safe_float(row.get("price_change_percentage_24h_in_currency")),
            mcap=safe_float(row.get("market_cap")),
            vol24=safe_float(row.get("total_volume")),
        )
    return out


def tr_and_atr_proxy(price: Optional[float], hi: Optional[float], lo: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return (true_range_24h, atr_proxy) using 24h high/low approximation."""
    if price is None or hi is None or lo is None:
        return None, None
    true_range = max(hi - lo, 0.0)
    atr = true_range  # one-day TR as ATR proxy
    return true_range, atr


def near_high_condition(price: Optional[float], hi: Optional[float], pct: float) -> bool:
    if price is None or hi is None or hi <= 0:
        return False
    return (hi - price) / hi * 100.0 <= pct


def range_pct(price: Optional[float], hi: Optional[float], lo: Optional[float]) -> Optional[float]:
    if price is None or hi is None or lo is None or price <= 0:
        return None
    return (max(hi - lo, 0.0) / price) * 100.0


def build_candidates(universe: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns: (candidates, debug_info)
    """
    cg_ids = list(universe.values())
    markets = fetch_markets(cg_ids)
    time.sleep(REQUEST_SLEEP_SEC)

    candidates: List[Dict[str, Any]] = []
    debugs: Dict[str, Any] = {"universe_count": len(universe), "scanned": [], "skipped": []}

    id_by_symbol = {sym: cid for sym, cid in universe.items()}

    for sym, cid in id_by_symbol.items():
        row = markets.get(cid)
        if not row:
            debugs["skipped"].append({"symbol": sym, "id": cid, "reason": "no_market_row"})
            continue

        tr24, atr = tr_and_atr_proxy(row.price, row.high24, row.low24)
        r_pct = range_pct(row.price, row.high24, row.low24)

        # Filter: basic momentum & range
        if row.pct24 is None or r_pct is None:
            debugs["skipped"].append({"symbol": sym, "id": cid, "reason": "insufficient_data"})
            continue

        if row.pct24 < MIN_PCT_24H or r_pct < MIN_RANGE_PCT:
            debugs["skipped"].append(
                {
                    "symbol": sym,
                    "id": cid,
                    "reason": "weak_momentum_or_range",
                    "pct24": row.pct24,
                    "range_pct": r_pct,
                }
            )
            continue

        near_hi = near_high_condition(row.price, row.high24, NEAR_HIGH_PCT)
        score = (row.pct24 or 0.0) + (r_pct or 0.0) + (5.0 if near_hi else 0.0)

        # Entry/Stop/T1/T2 using 24h high/low heuristic
        entry = row.price
        stop = row.low24  # “breakout bar low” proxy using 24h low
        t1 = t2 = None
        if atr is not None and entry is not None:
            t1 = entry + 0.8 * atr
            t2 = entry + 1.5 * atr

        candidates.append(
            {
                "symbol": sym,
                "id": cid,
                "price": row.price,
                "high24": row.high24,
                "low24": row.low24,
                "pct24": row.pct24,
                "range_pct": r_pct,
                "atr_proxy": atr,
                "near_24h_high": near_hi,
                "entry": entry,
                "stop": stop,
                "T1": t1,
                "T2": t2,
                "score": round(score, 4),
            }
        )
        debugs["scanned"].append({"symbol": sym, "id": cid, "ok": True})

    # Sort by score desc, then pct24
    candidates.sort(key=lambda d: (d.get("score") or 0.0, d.get("pct24") or 0.0), reverse=True)
    return candidates, debugs


def best_breakout(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Strong breakout if:
      - near_24h_high True
      - range_pct >= MIN_RANGE_PCT
      - pct24 >= (MIN_PCT_24H + 3)
    Returns the top candidate that meets this, else None.
    """
    for c in candidates:
        if c.get("near_24h_high") and (c.get("range_pct") or 0) >= MIN_RANGE_PCT and (c.get("pct24") or 0) >= (MIN_PCT_24H + 3.0):
            return c
    return None


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def main() -> None:
    started = time.time()
    status = "ok"
    errors: List[str] = []
    universe = dict(DEFAULT_UNIVERSE)

    # Optional: load an override universe file if present
    override_path = Path("data/universe.json")
    if override_path.exists():
        try:
            user_map = json.loads(override_path.read_text())
            if isinstance(user_map, dict):
                # normalize keys to SYMBOL upper
                clean: Dict[str, str] = {}
                for k, v in user_map.items():
                    if isinstance(k, str) and isinstance(v, str) and v.strip():
                        clean[k.upper()] = v.strip()
                if clean:
                    universe = clean
        except Exception as e:
            errors.append(f"universe_override_load_failed: {e!r}")

    # Run scan
    candidates: List[Dict[str, Any]] = []
    debug_info: Dict[str, Any] = {}
    try:
        candidates, debug_info = build_candidates(universe)
    except Exception as e:
        status = "degraded"
        errors.append(f"scan_failed: {e!r}")

    # Build signals object
    signals: Dict[str, Any] = {"type": "C", "note": "candidates", "count": len(candidates)}
    pick = best_breakout(candidates)
    if pick:
        signals = {
            "type": "B",
            "ticker": pick["symbol"],
            "entry": pick.get("entry"),
            "stop": pick.get("stop"),
            "T1": pick.get("T1"),
            "T2": pick.get("T2"),
            "atr_proxy": pick.get("atr_proxy"),
            "reason": "Strong breakout: near 24h high + momentum + range",
        }

    # Market snapshot (top N for quick view)
    snapshot = {
        "ts_utc": _now_iso(),
        "universe_size": len(universe),
        "candidates_top5": candidates[:5],
    }

    # Debug + stats
    run_stats = {
        "started_utc": datetime.utcfromtimestamp(started).isoformat() + "Z",
        "ended_utc": _now_iso(),
        "duration_sec": round(time.time() - started, 3),
        "errors": errors,
    }

    debug_blob = {
        "params": {
            "vs_currency": VS_CCY,
            "NEAR_HIGH_PCT": NEAR_HIGH_PCT,
            "MIN_RANGE_PCT": MIN_RANGE_PCT,
            "MIN_PCT_24H": MIN_PCT_24H,
        },
        "universe": universe,
        "debug": debug_info,
    }

    # Write files
    try:
        write_json(OUT_DIR / "market_snapshot.json", snapshot)
        write_json(OUT_DIR / "debug_scan.json", debug_blob)
        write_json(OUT_DIR / "run_stats.json", run_stats)
        write_json(OUT_DIR / "signals.json", signals)

        # Summary: minimal, used by your chat fetcher
        summary = {
            "status": status,
            "ts_utc": _now_iso(),
            "signals": signals,
            "candidates": candidates[:10],  # include a short list
        }
        write_json(OUT_DIR / "summary.json", summary)
    except Exception as e:
        # As a last resort, write a tiny summary so the workflow step still succeeds
        fallback = {
            "status": "degraded",
            "ts_utc": _now_iso(),
            "error": f"write_failed: {e!r}",
            "signals": {"type": "C", "count": 0},
            "candidates": [],
        }
        OUT_DIR.joinpath("summary.json").write_text(json.dumps(fallback, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()