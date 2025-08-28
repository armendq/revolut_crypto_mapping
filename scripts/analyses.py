#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses runner

- Fetches summary JSON (2 attempts, 60s wait between) from:
  https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json
- If both attempts fail -> prints:  "Hold and wait. Fetch failed twice."
- If JSON is obtained -> prints concise, actionable instructions per rules:
    * If signals.type == "B": output exact trade plan:
        - ticker, entry, stop (use breakout bar low), T1 (+0.8 ATR), T2 (+1.5 ATR),
          USD position size (prefer field from JSON; else compute).
        - Trail after +1R.
        - Respect: DOT is staked/untouchable; ETH and DOT are not rotated.
    * If signals.type == "C":
        - If candidates non-empty: list top candidates with entry/stop/sizing (same rules;
          exclude DOT/ETH from rotation list).
        - Else: "Hold and wait."
- Uses Europe/Prague time in the printed text.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import requests
from zoneinfo import ZoneInfo


SUMMARY_URL = "https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json"
RETRIES = 2
WAIT_SECONDS = 60

RISK_PCT = 0.012  # 1.2%
CAP_OK = 0.60     # 60% if regime.ok
CAP_BAD = 0.30    # 30% otherwise

TZ = ZoneInfo("Europe/Prague")

# ---------- Utilities ----------

def fetch_json_with_retry(url: str, retries: int = RETRIES, wait_s: int = WAIT_SECONDS) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                try:
                    return r.json()
                except json.JSONDecodeError:
                    pass  # fall through to retry
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(wait_s)
    return None


def fmt_price(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    # choose a sensible precision for crypto quotes
    if x >= 100:
        return f"{x:.2f}"
    if x >= 10:
        return f"{x:.3f}"
    if x >= 1:
        return f"{x:.4f}"
    if x >= 0.1:
        return f"{x:.5f}"
    if x >= 0.01:
        return f"{x:.6f}"
    return f"{x:.8f}"


def now_prg() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")


# ---------- Sizing ----------

@dataclass
class SizingInputs:
    entry: float
    stop: float
    equity: float
    cash: float
    regime_ok: bool


def compute_position_usd(si: SizingInputs) -> float:
    """
    Risk-based USD position sizing:
      - Risk = equity * 1.2%
      - Units = Risk / (entry - stop)
      - USD = min(Units*entry, cap*equity, cash)
      - Cap = 60% equity if regime_ok else 30%
    """
    if si.entry is None or si.stop is None or si.entry <= 0:
        return 0.0
    stop_dist = si.entry - si.stop
    if stop_dist <= 0:
        return 0.0
    risk_usd = si.equity * RISK_PCT
    units = risk_usd / stop_dist
    gross_usd = units * si.entry
    cap = CAP_OK if si.regime_ok else CAP_BAD
    max_alloc = cap * si.equity
    # Respect available cash as well
    return max(0.0, min(gross_usd, max_alloc, si.cash))


# ---------- Parsing helpers ----------

def read_float(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def pick_stop(payload: Dict[str, Any]) -> Optional[float]:
    # Prefer explicit breakout bar low if present
    for key in ("breakout_low", "stop", "breakout_bar_low", "low"):
        v = read_float(payload, key)
        if v is not None:
            return v
    # Fallback: entry - ATR (very conservative)
    entry = read_float(payload, "entry", "breakout_entry", "price")
    atr = read_float(payload, "atr")
    if entry is not None and atr is not None and atr > 0:
        return entry - atr
    return None


def pick_entry(payload: Dict[str, Any]) -> Optional[float]:
    return read_float(payload, "entry", "breakout_entry", "price")


def pick_atr(payload: Dict[str, Any]) -> Optional[float]:
    return read_float(payload, "atr", "ATR", "atr_14")


def get_equity_cash(payload: Dict[str, Any]) -> Tuple[float, float]:
    eq = read_float(payload, "equity") or 0.0
    cash = read_float(payload, "cash") or 0.0
    return float(eq), float(cash)


def is_regime_ok(payload: Dict[str, Any]) -> bool:
    regime = payload.get("regime") or {}
    ok = regime.get("ok")
    return bool(ok)


# ---------- Main rendering ----------

def render_type_b(data: Dict[str, Any]) -> str:
    sig = data.get("signals", {}) or {}
    payload = sig.get("payload") or {}
    ticker = (sig.get("ticker") or payload.get("ticker") or "").upper()

    # Respect DOT/ETH constraints
    if ticker in ("DOT", "POLKADOT"):
        return "Hold and wait. DOT is staked and untouchable."

    entry = pick_entry(payload)
    stop = pick_stop(payload)
    atr = pick_atr(payload)

    # Targets
    t1 = (entry + 0.8 * atr) if (entry is not None and atr is not None) else None
    t2 = (entry + 1.5 * atr) if (entry is not None and atr is not None) else None

    # Sizing
    position_usd = read_float(payload, "position_usd", "usd_size", "size_usd")
    if position_usd is None:
        equity, cash = get_equity_cash(data)
        regime_ok = is_regime_ok(data)
        si = SizingInputs(
            entry=float(entry) if entry is not None else 0.0,
            stop=float(stop) if stop is not None else 0.0,
            equity=float(equity),
            cash=float(cash),
            regime_ok=regime_ok,
        )
        position_usd = compute_position_usd(si)

    ts = now_prg()
    # Compose concise instruction
    lines = [
        f"[{ts} Europe/Prague]",
        f"BUY {ticker} @ {fmt_price(entry)}",
        f"Stop: {fmt_price(stop)} (breakout bar low).",
        f"T1: {fmt_price(t1)} (+0.8 ATR), T2: {fmt_price(t2)} (+1.5 ATR).",
        f"Position: ${fmt_price(position_usd)}.",
        "After +1R, trail stop.",
    ]
    return " ".join(lines)


def render_type_c(data: Dict[str, Any]) -> str:
    sig = data.get("signals", {}) or {}
    candidates: List[Dict[str, Any]] = sig.get("candidates") or []
    if not candidates:
        return "Hold and wait."

    equity, cash = get_equity_cash(data)
    regime_ok = is_regime_ok(data)

    rendered = []
    for c in candidates:
        ticker = (c.get("ticker") or "").upper()
        if ticker in ("ETH", "DOT", "POLKADOT"):
            # ETH and DOT are not rotated -> skip from candidate rotation list
            continue

        entry = pick_entry(c)
        stop = pick_stop(c)
        atr = pick_atr(c)
        t1 = (entry + 0.8 * atr) if (entry is not None and atr is not None) else None
        t2 = (entry + 1.5 * atr) if (entry is not None and atr is not None) else None

        position_usd = read_float(c, "position_usd", "usd_size", "size_usd")
        if position_usd is None:
            si = SizingInputs(
                entry=float(entry) if entry is not None else 0.0,
                stop=float(stop) if stop is not None else 0.0,
                equity=float(equity),
                cash=float(cash),
                regime_ok=regime_ok,
            )
            position_usd = compute_position_usd(si)

        rendered.append(
            f"{ticker}: entry {fmt_price(entry)}, stop {fmt_price(stop)}, "
            f"T1 {fmt_price(t1)}, T2 {fmt_price(t2)}, size ${fmt_price(position_usd)}"
        )

    if not rendered:
        return "Hold and wait."
    ts = now_prg()
    prefix = f"[{ts} Europe/Prague] Rotation candidates:"
    return prefix + " " + " | ".join(rendered)


def main() -> None:
    data = fetch_json_with_retry(SUMMARY_URL, retries=RETRIES, wait_s=WAIT_SECONDS)
    if data is None:
        print("Hold and wait. Fetch failed twice.")
        return

    signals = data.get("signals") or {}
    sig_type = (signals.get("type") or "").upper()

    if sig_type == "B":
        out = render_type_b(data)
    elif sig_type == "C":
        out = render_type_c(data)
    else:
        out = "Hold and wait."

    # One concise, actionable line; no questions.
    print(out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail safe: never crash the workflow with a stack trace — print a hold line instead.
        print("Hold and wait. Fetch failed twice.")
        # Optionally log to stderr for debugging without breaking the contract:
        print(f"# Debug: {e}", file=sys.stderr)