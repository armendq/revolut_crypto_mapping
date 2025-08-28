#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses runner:
- Modes: deep | light-hourly  (both run the same fetch/format logic)
- Fetches summary.json from the repo (retry once after 60s on non-200/invalid JSON)
- Emits concise trade instructions per requirements
- Writes timestamped + latest copies under public_runs/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # Fallback

import urllib.request
import urllib.error


SUMMARY_URL = "https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json"

# ---------- utils

def prague_now_str():
    if ZoneInfo:
        tz = ZoneInfo("Europe/Prague")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    # Fallback to UTC label if zoneinfo missing
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def fetch_json_with_retry(url: str, wait_seconds: int = 60, max_attempts: int = 2):
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                code = resp.getcode()
                if code != 200:
                    raise urllib.error.HTTPError(url, code, f"HTTP {code}", hdrs=None, fp=None)
                raw = resp.read()
                data = json.loads(raw.decode("utf-8"))
                return data
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_attempts:
                # explicit wait only inside the script; workflow logs will show the pause
                time.sleep(wait_seconds)
    return None


def ensure_dirs():
    Path("public_runs/latest").mkdir(parents=True, exist_ok=True)


def write_json_files(payload: dict):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = Path("public_runs")
    ts_dir = base / ts
    ts_dir.mkdir(parents=True, exist_ok=True)
    latest = base / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (ts_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (latest / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------- sizing

def _num(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def cap_pct(regime_ok: bool):
    return 0.60 if regime_ok else 0.30


def compute_position_usd(entry: float, stop: float, equity: float, cash: float, regime_ok: bool, risk_pct: float = 0.012):
    if entry is None or stop is None or entry <= 0:
        return None
    risk_dollars = max(equity * risk_pct, 0.0)
    stop_dist = max(entry - stop, 0.0)
    if stop_dist <= 0:
        return None

    # theoretical position to risk 1.2% per trade
    theoretical_size = risk_dollars / stop_dist * entry  # convert units→USD by multiplying by entry
    # cap by regime & cash
    abs_cap = equity * cap_pct(regime_ok)
    size_cap = min(abs_cap, cash if cash is not None else abs_cap)
    final_size = min(theoretical_size, size_cap)
    return max(round(final_size, 2), 0.0)


# ---------- formatting

def print_hold_failed_twice():
    print('Hold and wait. Fetch failed twice.')


def fmt_trade_line(ticker, entry, stop, atr, pos_usd):
    # Targets per rule: T1 +0.8 ATR, T2 +1.5 ATR; trail after +1R (mention once)
    t1 = entry + 0.8 * atr if entry is not None and atr is not None else None
    t2 = entry + 1.5 * atr if entry is not None and atr is not None else None

    def f(x):
        return f"{x:.4f}" if x is not None else "n/a"

    return (f"{ticker}: entry {f(entry)}, stop {f(stop)}, "
            f"T1 {f(t1)}, T2 {f(t2)}, size ${pos_usd:,.0f}")


def is_dot_or_eth(ticker: str):
    if not ticker:
        return False
    t = ticker.upper()
    return "DOT" in t or t == "ETH" or t.endswith("ETH")


# ---------- main logic based on your summary.json spec

def handle_payload(payload: dict):
    # Persist a copy so Pages always has a file (even on Hold).
    try:
        write_json_files(payload)
    except Exception:
        pass

    regime = payload.get("regime", {})
    regime_ok = bool(regime.get("ok", False))

    equity = _num(payload.get("equity"), 0.0)
    cash = _num(payload.get("cash"), equity)  # if not provided, assume fully available

    sig = payload.get("signals") or {}
    stype = (sig.get("type") or "").upper()

    now_str = prague_now_str()

    # Type B: single trade instruction
    if stype == "B":
        ticker = (sig.get("ticker") or sig.get("symbol") or "").upper()

        # Respect DOT staked + ETH/DOT not rotated
        if "DOT" in ticker:
            print(f"[{now_str} Europe/Prague] Hold and wait. DOT is staked/untouchable.")
            return

        entry = _num(sig.get("entry") or sig.get("price") or sig.get("breakout"))
        atr = _num(sig.get("atr") or sig.get("ATR"))
        stop = _num(sig.get("stop") or sig.get("breakout_low") or (entry - atr if (entry and atr) else None))

        # Provided position size takes precedence
        pos_usd = _num(sig.get("position_usd") or sig.get("size_usd"))
        if pos_usd is None:
            pos_usd = compute_position_usd(entry, stop, equity, cash, regime_ok)

        if pos_usd is None or entry is None or stop is None or atr is None:
            print(f"[{now_str} Europe/Prague] Hold and wait.")
            return

        line = fmt_trade_line(ticker, entry, stop, atr, pos_usd)
        print(f"[{now_str} Europe/Prague] TRADE — trail after +1R.\n{line}")
        return

    # Type C: candidates list
    if stype == "C":
        cands = payload.get("candidates") or []
        out_lines = []

        for c in cands:
            ticker = (c.get("ticker") or c.get("symbol") or "").upper()
            if "DOT" in ticker:
                continue  # untouchable
            entry = _num(c.get("entry") or c.get("price") or c.get("breakout"))
            atr = _num(c.get("atr") or c.get("ATR"))
            stop = _num(c.get("stop") or c.get("breakout_low") or (entry - atr if (entry and atr) else None))
            pos_usd = _num(c.get("position_usd") or c.get("size_usd"))
            if pos_usd is None:
                pos_usd = compute_position_usd(entry, stop, equity, cash, regime_ok)
            if entry is None or stop is None or atr is None or pos_usd is None:
                continue
            out_lines.append(fmt_trade_line(ticker, entry, stop, atr, pos_usd))

        if out_lines:
            print(f"[{prague_now_str()} Europe/Prague] CANDIDATES — trail after +1R.\n" + "\n".join(out_lines))
        else:
            print(f"[{prague_now_str()} Europe/Prague] Hold and wait.")
        return

    # Anything else
    print(f"[{now_str} Europe/Prague] Hold and wait.")


def main():
    parser = argparse.ArgumentParser(description="Analyses runner")
    parser.add_argument("--mode", choices=["deep", "light-hourly"], default="deep")
    parser.add_argument("--url", default=SUMMARY_URL, help="summary.json URL")
    args = parser.parse_args()

    # Make sure output tree exists so later steps don't fail
    ensure_dirs()

    # Fetch summary with retry
    payload = fetch_json_with_retry(args.url, wait_seconds=60, max_attempts=2)
    if payload is None:
        print_hold_failed_twice()
        # still write a minimal file so Pages has something
        try:
            write_json_files({"error": "fetch_failed_twice", "at": datetime.utcnow().isoformat()})
        except Exception:
            pass
        sys.exit(0)

    # Process & print
    try:
        handle_payload(payload)
    except Exception as e:
        # Be resilient: never crash the workflow; just fall back to Hold
        print(f"[{prague_now_str()} Europe/Prague] Hold and wait.")
        # best-effort persistence of raw payload for debugging
        try:
            write_json_files({"error": f"exception: {type(e).__name__}", "detail": str(e), "raw": payload})
        except Exception:
            pass


if __name__ == "__main__":
    main()