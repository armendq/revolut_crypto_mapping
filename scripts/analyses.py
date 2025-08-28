#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyses.py
- Fetches latest summary.json (2 attempts, 60s wait between).
- Accepts --mode (deep / light-fast / light-hourly) to stay compatible with the workflow, but ignores it.
- Emits concise trading instructions based on signals:
    * If signals.type == "B": print exact trade plan (ticker, entry, stop, T1, T2, USD size).
    * If signals.type == "C": list candidates with entry/stop/sizing; else say Hold and wait.
    * Fallback: "Hold and wait." on fetch/parse problems or no actionable signals.

Business rules:
- Use Europe/Prague time in the output.
- If position size is provided in JSON, use it. Otherwise compute:
    size_cap = 0.60 if regime.ok else 0.30
    RISK_PCT = 0.012 (1.2% of equity)
    risk_dollars = equity * RISK_PCT
    unit_risk = abs(entry - stop)
    position_usd = min(risk_dollars / unit_risk * entry, (equity + cash) * size_cap)
- Targets: T1 = entry + 0.8 * ATR; T2 = entry + 1.5 * ATR (when ATR provided).
- Stop = breakout bar low (when provided; otherwise 'stop' if present).
- ETH and DOT are not rotated; DOT is staked/untouchable: skip DOT trades.
- Trail after +1R (echoed in text).

Output is intentionally concise/actionable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

import urllib.request
import urllib.error

DEFAULT_URL = "https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json"
RISK_PCT = 0.012  # 1.2%


def now_prague_str() -> str:
    tz = ZoneInfo("Europe/Prague") if ZoneInfo else None
    dt = datetime.utcnow() if tz is None else datetime.now(tz)
    if tz is None:
        # Mark as UTC if zoneinfo is unavailable
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def fetch_json(url: str, attempts: int = 2, wait_seconds: int = 60) -> Optional[Dict[str, Any]]:
    last_err: Optional[str] = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "analyses-bot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    last_err = f"HTTP {resp.status}"
                    raise urllib.error.HTTPError(url, resp.status, "non-200", resp.headers, None)
                data = resp.read()
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            last_err = str(e)
            if i + 1 < attempts:
                time.sleep(wait_seconds)
    # Final failure
    print("Hold and wait. Fetch failed twice.", flush=True)
    if last_err:
        print(f"# debug: fetch error -> {last_err}", file=sys.stderr)
    return None


def _get_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def first_nonempty(*keys: str, obj: Dict[str, Any]) -> Any:
    for k in keys:
        if k in obj and obj[k] not in (None, "", []):
            return obj[k]
    return None


@dataclass
class Account:
    equity: float
    cash: float
    regime_ok: bool


def read_account(root: Dict[str, Any]) -> Account:
    meta = root.get("meta") or root.get("account") or {}
    equity = _get_float(first_nonempty("equity", "balance", obj=meta)) or 0.0
    cash = _get_float(meta.get("cash")) or 0.0
    regime = root.get("regime") or meta.get("regime") or {}
    regime_ok = bool(regime.get("ok", True))
    return Account(equity=equity, cash=cash, regime_ok=regime_ok)


def compute_position_usd(entry: Optional[float], stop: Optional[float], acc: Account,
                         provided: Optional[float]) -> Optional[float]:
    if provided is not None:
        return max(0.0, provided)
    if entry is None or stop is None or acc.equity <= 0:
        return None
    unit_risk = abs(entry - stop)
    if unit_risk <= 0:
        return None
    risk_dollars = acc.equity * RISK_PCT
    raw_units = risk_dollars / unit_risk
    raw_usd = raw_units * entry
    cap = 0.60 if acc.regime_ok else 0.30
    max_usd = (acc.equity + acc.cash) * cap
    return max(0.0, min(raw_usd, max_usd))


def format_usd(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.0f}" if x >= 1000 else f"${x:,.2f}"


def build_targets(entry: Optional[float], atr: Optional[float]) -> (str, str):
    if entry is None or atr is None:
        return "n/a", "n/a"
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr
    return f"{t1:.6g}", f"{t2:.6g}"


def normalize_symbol(sym: Optional[str]) -> Optional[str]:
    if not sym:
        return None
    s = str(sym).upper().strip()
    # common mappings like "BTCUSDT" -> "BTC"
    if s.endswith("USDT"):
        s = s[:-4]
    if s.endswith("USD"):
        s = s[:-3]
    return s


def is_dot(sym: Optional[str]) -> bool:
    s = normalize_symbol(sym)
    return s in {"DOT", "DOT2"}  # be forgiving


def is_eth(sym: Optional[str]) -> bool:
    s = normalize_symbol(sym)
    return s == "ETH"


def handle_type_b(root: Dict[str, Any]) -> str:
    sig = root.get("signals") or {}
    trade = sig.get("trade") or sig.get("signal") or root.get("trade") or sig
    # Extract fields with defensive fallbacks
    sym = normalize_symbol(first_nonempty("ticker", "symbol", "asset", obj=trade))
    if is_dot(sym):
        return f"[{now_prague_str()}] Hold and wait. DOT is staked/untouchable."
    entry = _get_float(first_nonempty("entry", "price", "trigger", obj=trade))
    atr = _get_float(first_nonempty("atr", "ATR", obj=trade))
    stop = _get_float(first_nonempty("breakout_bar_low", "breakout_low", "stop", obj=trade))
    provided_size = _get_float(first_nonempty("position_usd", "size_usd", "usd_size", obj=trade))
    acc = read_account(root)
    size_usd = compute_position_usd(entry, stop, acc, provided_size)
    t1, t2 = build_targets(entry, atr)

    parts: List[str] = []
    parts.append(f"[{now_prague_str()}] BREAKOUT BUY")
    sym_show = sym or "UNKNOWN"
    entry_s = "n/a" if entry is None else f"{entry:.6g}"
    stop_s = "n/a" if stop is None else f"{stop:.6g}"
    parts.append(f"{sym_show} | Entry {entry_s} | Stop {stop_s} | T1 {t1} | T2 {t2} | Size {format_usd(size_usd)}")
    if is_eth(sym):
        parts.append("Note: ETH not rotated. Trail after +1R.")
    else:
        parts.append("Trail after +1R.")
    return "\n".join(parts)


def handle_type_c(root: Dict[str, Any]) -> str:
    sig = root.get("signals") or {}
    cands = sig.get("candidates") or sig.get("watchlist") or []
    if not isinstance(cands, list) or len(cands) == 0:
        return "Hold and wait."
    acc = read_account(root)

    lines: List[str] = [f"[{now_prague_str()}] CANDIDATES"]
    shown = 0
    for c in cands:
        sym = normalize_symbol(first_nonempty("ticker", "symbol", "asset", obj=c))
        if is_dot(sym):
            # Skip DOT entirely as per constraints
            continue
        entry = _get_float(first_nonempty("entry", "price", "trigger", obj=c))
        atr = _get_float(first_nonempty("atr", "ATR", obj=c))
        stop = _get_float(first_nonempty("breakout_bar_low", "breakout_low", "stop", obj=c))
        provided_size = _get_float(first_nonempty("position_usd", "size_usd", "usd_size", obj=c))
        size_usd = compute_position_usd(entry, stop, acc, provided_size)
        t1, t2 = build_targets(entry, atr)

        entry_s = "n/a" if entry is None else f"{entry:.6g}"
        stop_s = "n/a" if stop is None else f"{stop:.6g}"
        sym_show = sym or "UNKNOWN"
        lines.append(f"{sym_show} | Entry {entry_s} | Stop {stop_s} | T1 {t1} | T2 {t2} | Size {format_usd(size_usd)}")
        shown += 1
        if shown >= 6:  # keep concise
            break

    if shown == 0:
        return "Hold and wait."
    lines.append("Trail after +1R. (ETH/DOT not rotated; DOT excluded.)")
    return "\n".join(lines)


def render(root: Dict[str, Any]) -> str:
    signals = root.get("signals") or {}
    stype = str(signals.get("type") or root.get("type") or "").upper()
    if stype == "B":
        return handle_type_b(root)
    if stype == "C":
        return handle_type_c(root)
    return "Hold and wait."


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate trading instructions from summary.json")
    ap.add_argument("--url", default=os.getenv("SUMMARY_URL", DEFAULT_URL),
                    help="Override summary.json URL")
    # Accept --mode to remain compatible with existing workflow; we don't branch on it for now.
    ap.add_argument("--mode", default=None,
                    help="Run mode (deep, light-fast, light-hourly). Accepted but ignored.")
    args = ap.parse_args(argv)

    data = fetch_json(args.url, attempts=2, wait_seconds=60)
    if not data:
        # Message already printed on failure
        return 0

    out = render(data)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())