#!/usr/bin/env python3
"""
analyses.py

Reads summary.json from the public_runs/latest folder and prints concise,
actionable trading instructions based on the user's rules.

Rules implemented
-----------------
- Fetch URL twice at most; on first failure wait 60s then retry.
  If both fail -> print exactly: 'Hold and wait. Fetch failed twice.'
- If signals.type == "B":
    Output exact trade instructions with:
      ticker, entry, stop (breakout bar low), T1 (+0.8 ATR), T2 (+1.5 ATR),
      position USD (use provided size if present; else compute).
    Trail after +1R note is appended.
- If signals.type == "C" and candidates non-empty:
    List top candidates with entry/stop/position, same sizing rules.
- Else: print "Hold and wait."
- ETH and DOT are not rotated; DOT is staked/untouchable -> never propose trades in DOT.
- Europe/Prague time is shown in the first line.
- Output stays concise and human-executable.

Environment / CLI
-----------------
- SUMMARY_URL env var (or --url) can override the default URL.
- DEBUG=1 env var prints diagnostic info to stderr (never to stdout).
- RUN_MODE may be set externally (not required by this script, but respected
  if other tooling passes it).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    # Python 3.9+ only
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

DEFAULT_URL = "https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json"
RISK_PCT_DEFAULT = 0.012  # 1.2% per trade risk


# ------------------------- utilities ------------------------- #

def log(msg: str) -> None:
    """Debug logs (stderr only)."""
    if os.getenv("DEBUG"):
        print(msg, file=sys.stderr)


def now_prague_str() -> str:
    """Return current time in Europe/Prague."""
    try:
        if ZoneInfo is None:
            raise RuntimeError("zoneinfo not available")
        tz = ZoneInfo("Europe/Prague")
        from datetime import datetime
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        # Fallback: local time with '(local)' marker if zoneinfo missing
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M (local)")


def fetch_with_retry(url: str, attempts: int = 2, wait: int = 60) -> Dict[str, Any]:
    """Fetch JSON from URL with retries and a blocking wait between attempts."""
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            log(f"[fetch] attempt {i+1} -> {url}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                status = getattr(resp, "status", getattr(resp, "code", None))
                if status != 200:
                    raise RuntimeError(f"HTTP {status}")
                raw = resp.read()
            data = json.loads(raw)
            log(f"[fetch] OK 200, {len(raw)} bytes")
            return data
        except Exception as e:  # noqa: PERF203
            last_err = e
            log(f"[fetch] failed: {e!s}")
            if i < attempts - 1:
                log(f"[fetch] waiting {wait}s before retry…")
                time.sleep(wait)

    raise RuntimeError(f"Fetch failed twice: {last_err}")


def first_non_null(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default


# ------------------------- sizing logic ------------------------- #

@dataclass
class AccountContext:
    equity: Optional[float]
    cash: Optional[float]
    regime_ok: bool
    risk_pct: float = RISK_PCT_DEFAULT
    cap_ok: float = 0.60
    cap_bad: float = 0.30

    @property
    def cap(self) -> float:
        return self.cap_ok if self.regime_ok else self.cap_bad


@dataclass
class Instrument:
    ticker: str
    entry: float
    stop: float
    atr: Optional[float] = None
    size_usd_hint: Optional[float] = None

    def t1(self) -> Optional[float]:
        if self.atr is None:
            return None
        return self.entry + 0.8 * self.atr

    def t2(self) -> Optional[float]:
        if self.atr is None:
            return None
        return self.entry + 1.5 * self.atr


def clamp_position(ctx: AccountContext, desired_usd: float) -> float:
    """Clamp position by available cash and exposure cap if provided."""
    if desired_usd <= 0 or not math.isfinite(desired_usd):
        return 0.0

    # Exposure cap in USD if equity known.
    exposure_cap_usd = None
    if ctx.equity is not None:
        exposure_cap_usd = ctx.equity * ctx.cap

    # Available cash if known.
    cash_limit = ctx.cash if ctx.cash is not None else None

    capped = desired_usd
    if exposure_cap_usd is not None:
        capped = min(capped, exposure_cap_usd)
    if cash_limit is not None:
        capped = min(capped, cash_limit)

    return max(0.0, float(capped))


def compute_position_usd(ctx: AccountContext, inst: Instrument) -> float:
    """Return position size in USD using either given hint or risk model."""
    # Use hint if present and valid (already computed upstream)
    if inst.size_usd_hint is not None and inst.size_usd_hint > 0:
        return clamp_position(ctx, inst.size_usd_hint)

    # Risk model: risk_dollars = equity * RISK_PCT
    if ctx.equity is None:
        # If no equity data, fall back to using cash with the cap
        if ctx.cash is None:
            return 0.0
        equity_proxy = ctx.cash / ctx.cap if ctx.cap > 0 else ctx.cash
        eq = max(0.0, equity_proxy)
    else:
        eq = max(0.0, ctx.equity)

    risk_dollars = eq * ctx.risk_pct
    per_unit_risk = max(1e-12, inst.entry - inst.stop)  # never divide by 0/neg

    units = risk_dollars / per_unit_risk
    desired_usd = units * inst.entry
    return clamp_position(ctx, desired_usd)


# ------------------------- JSON parsing helpers ------------------------- #

def parse_ctx(j: Dict[str, Any]) -> AccountContext:
    regime_ok = bool(first_non_null(
        j.get("regime", {}).get("ok"),
        j.get("signals", {}).get("regime_ok"),
        True  # default to OK if not present
    ))
    equity = j.get("account", {}).get("equity")
    cash = j.get("account", {}).get("cash")
    risk_pct = float(first_non_null(j.get("config", {}).get("risk_pct"), RISK_PCT_DEFAULT))

    return AccountContext(equity=equity, cash=cash, regime_ok=regime_ok, risk_pct=risk_pct)


def to_instrument(d: Dict[str, Any]) -> Optional[Instrument]:
    """Build Instrument from a dict, being liberal with field names."""
    ticker = first_non_null(d.get("ticker"), d.get("symbol"), d.get("pair"))
    if not ticker or not isinstance(ticker, str):
        return None

    # Respect DOT rule: never produce trades in DOT
    sym = ticker.upper().replace("USDT", "").replace("/", "").strip()
    if sym in {"DOT"}:
        return None

    entry = first_non_null(d.get("entry"), d.get("breakout"), d.get("price"))
    stop = first_non_null(d.get("stop"), d.get("breakout_low"), d.get("sl"))
    atr = first_non_null(d.get("atr"), d.get("ATR"), d.get("atr14"))
    size_hint = first_non_null(d.get("position_usd"), d.get("size_usd"), d.get("usd_size"))

    try:
        entry = float(entry)
        stop = float(stop)
    except Exception:
        return None

    try:
        atr = float(atr) if atr is not None else None
    except Exception:
        atr = None

    try:
        size_hint = float(size_hint) if size_hint is not None else None
    except Exception:
        size_hint = None

    return Instrument(ticker=ticker, entry=entry, stop=stop, atr=atr, size_usd_hint=size_hint)


# ------------------------- output formatting ------------------------- #

def fmt_price(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x == 0:
        return "0"
    # Dynamic precision for crypto quotes
    mag = abs(x)
    if mag >= 100:
        return f"{x:,.2f}"
    if mag >= 1:
        return f"{x:,.4f}"
    if mag >= 0.01:
        return f"{x:,.6f}"
    return f"{x:,.8f}"


def instruction_line(inst: Instrument, pos_usd: float) -> str:
    t1 = inst.t1()
    t2 = inst.t2()
    return (
        f"{inst.ticker}: entry {fmt_price(inst.entry)}, "
        f"stop {fmt_price(inst.stop)}, "
        f"T1 {fmt_price(t1)}, T2 {fmt_price(t2)}, "
        f"size ${pos_usd:,.0f}"
    )


def print_header() -> None:
    print(f"[{now_prague_str()} Europe/Prague]")


# ------------------------- main decision logic ------------------------- #

def handle_type_B(j: Dict[str, Any]) -> None:
    ctx = parse_ctx(j)

    # possible shapes: signals.trades (list) or signals (single)
    sig = j.get("signals", {})
    items: List[Dict[str, Any]] = []
    if isinstance(sig.get("trades"), list):
        items = sig["trades"]
    else:
        # If single signal dict contains a trade-like object
        items = [sig]

    # filter out ETH/DOT rotation rules:
    valid_instruments: List[Instrument] = []
    for d in items:
        inst = to_instrument(d)
        if not inst:
            continue
        sym = inst.ticker.upper()
        # DOT never, ETH allowed only if it's not a rotation out/instruction
        if sym.startswith("ETH"):
            # If JSON explicitly says it's a rotation (best-effort), skip
            if str(d.get("action", "")).lower() in {"rotate", "rotate_out"}:
                continue
        valid_instruments.append(inst)

    if not valid_instruments:
        print("Hold and wait.")
        return

    print_header()
    for inst in valid_instruments:
        pos = compute_position_usd(ctx, inst)
        line = instruction_line(inst, pos)
        print(line)
    print("Use breakout bar low as stop; trail after +1R.")


def handle_type_C(j: Dict[str, Any]) -> None:
    ctx = parse_ctx(j)
    cands = first_non_null(
        j.get("candidates"),
        j.get("signals", {}).get("candidates"),
        []
    )

    instruments: List[Instrument] = []
    for d in cands or []:
        inst = to_instrument(d or {})
        if not inst:
            continue
        sym = inst.ticker.upper()
        # ETH/DOT not rotated rule
        if sym.startswith("ETH"):
            action = str(d.get("action", "")).lower()
            if action in {"rotate", "rotate_out"}:
                continue
        instruments.append(inst)

    if not instruments:
        print("Hold and wait.")
        return

    print_header()
    print("Top candidates:")
    for inst in instruments:
        pos = compute_position_usd(ctx, inst)
        print(" - " + instruction_line(inst, pos))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate trading instructions from summary.json")
    ap.add_argument("--url", default=os.getenv("SUMMARY_URL", DEFAULT_URL),
                    help="Override summary.json URL")
    args = ap.parse_args(argv)

    try:
        j = fetch_with_retry(args.url, attempts=2, wait=60)
    except Exception:
        print("Hold and wait. Fetch failed twice.")
        return 0

    # Defensive checks
    signals = j.get("signals") or {}
    sig_type = str(first_non_null(signals.get("type"), j.get("type"), "")).upper()

    log(f"[json] signals.type={sig_type!r}")

    if sig_type == "B":
        handle_type_B(j)
    elif sig_type == "C":
        handle_type_C(j)
    else:
        print("Hold and wait.")

    return 0


if __name__ == "__main__":
    sys.exit(main())