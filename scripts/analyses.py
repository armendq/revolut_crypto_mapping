from scripts.marketdata import get_btc_5m_klines, ema, vwap
from pathlib import Path
import json

CFG_PATH = Path("data/portfolio.json")
PORT = {
    "equity_usd": None,
    "cash_usd": 0,
    "floor_usd": 36500,
    "floor_buffer_usd": 1000,
    "halt_new_entries_equity_usd": 37500,
    "exclude_from_rotation": ["ETH","DOT"],
    "holdings": []
}
if CFG_PATH.exists():
    PORT.update(json.loads(CFG_PATH.read_text()))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs the Aggressive Breakout + Micro Pullback scan on Revolut-listed coins
using open-source live data (Binance primary; CoinGecko fallback).

Outputs EXACTLY one of:
A) Preservation breach -> rotation/raise-cash instructions
B) Single best A+ setup (bullet list)
C) "Hold and wait."

Environment variables you can set in GitHub Action or locally:
- EQUITY_USD     : total account equity (float). Default 41000.
- CASH_USD       : free cash available (float). Default 32000.
- FLOOR_USD      : hard floor (default 36500).
- FLOOR_BUFFER   : required buffer above floor (default 1000).
- DAILY_PNL_PCT  : today’s realized PnL % (negative triggers loss cap). Default 0.0
- REGIME_SOURCE  : 'binance' (default)

This script prints the final decision to STDOUT (so Action logs show A/B/C).
It also writes `data/market_snapshot.json` and `data/signals.json`.
"""

import os, sys, json, time, math
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MAP_CSV = os.path.join(DATA_DIR, "revolut_mapping.csv")
SNAP_PATH = os.path.join(DATA_DIR, "market_snapshot.json")
SIG_PATH  = os.path.join(DATA_DIR, "signals.json")

# ----------- Config ----------
EQUITY_USD   = float(os.environ.get("EQUITY_USD", "41000"))
CASH_USD     = float(os.environ.get("CASH_USD", "32000"))
FLOOR_USD    = float(os.environ.get("FLOOR_USD", "36500"))
FLOOR_BUFFER = float(os.environ.get("FLOOR_BUFFER", "1000"))
DAILY_PNL_PCT= float(os.environ.get("DAILY_PNL_PCT", "0.0"))   # e.g., -1.2 for -1.2%
REGIME_SOURCE= os.environ.get("REGIME_SOURCE", "binance")

# ----------- Helper utils ----------
BINANCE = "https://api.binance.com"
CG = "https://api.coingecko.com/api/v3"

def now_utc():
    return datetime.now(timezone.utc)

def read_mapping():
    if not os.path.exists(MAP_CSV):
        raise FileNotFoundError("revolut_mapping.csv not found. Run mapping step first.")
    df = pd.read_csv(MAP_CSV)
    # expected columns: revolut_symbol, name, coingecko_id, binance_symbol, coinbase_symbol, kraken_symbol, bitstamp_symbol
    return df

def safe_get(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# ----------- Regime (BTC 5m) ----------
def binance_klines(symbol="BTCUSDT", interval="5m", limit=60):
    url = f"{BINANCE}/api/v3/klines"
    js = safe_get(url, {"symbol":symbol, "interval":interval, "limit":limit})
    return js or []

def vwap_from_klines(kl):
    # kl: [open_time, open, high, low, close, volume, ... quote_asset_volume]
    # approximate VWAP = sum(price_avg * volume) / sum(volume) over window
    if not kl: return None
    pv, v = 0.0, 0.0
    for k in kl:
        high, low, close, vol = float(k[2]), float(k[3]), float(k[4]), float(k[5])
        price_avg = (high + low + close) / 3.0
        pv += price_avg * vol
        v  += vol
    if v == 0: return None
    return pv / v

def ema(values, length=9):
    if len(values) < 1: return None
    s = pd.Series(values, dtype=float).ewm(span=length, adjust=False).mean()
    return float(s.iloc[-1])

def btc_regime():
    kl = binance_klines("BTCUSDT", "5m", 60)
    if len(kl) == 0: 
        return {"ok": False, "reason":"no-binance-klines"}
    closes = [float(k[4]) for k in kl]
    last_close = closes[-1]
    vwap = vwap_from_klines(kl[-60:])  # rolling approx
    ema9 = ema(closes, 9)
    strong = (last_close > (vwap or 0)) and (last_close > (ema9 or 0))
    return {"ok": True, "last_close":last_close, "vwap":vwap, "ema9":ema9, "strong":strong}

# ----------- Universe filter ----------
def binance_24h_ticker(symbol):
    js = safe_get(f"{BINANCE}/api/v3/ticker/24hr", {"symbol":symbol})
    return js

def binance_best_bid_ask(symbol):
    js = safe_get(f"{BINANCE}/api/v3/ticker/bookTicker", {"symbol":symbol})
    if js:
        return float(js.get("bidPrice", 0)), float(js.get("askPrice", 0))
    return None, None

def eligible_universe(df_map):
    """
    Universe: Revolut X coins w/ 24h volume ≥ $8M and spread ≤ 0.5%.
    Exclude ETH and staked DOT from rotation (but they may still be in portfolio).
    We’ll evaluate on Binance symbols when available, else drop for this run.
    """
    rows = []
    for _, r in df_map.iterrows():
        sym = str(r.get("revolut_symbol","")).upper()
        if sym in ("ETH","DOT.STAKED","STKDOT","DOTS","DOT-STAKE"): 
            continue
        b = str(r.get("binance_symbol") or "").upper().replace("-", "")
        if not b or not b.endswith("USDT"):
            # Try to guess: append USDT if pure ticker provided
            if b and "USDT" not in b and len(b) > 0:
                b = b + "USDT"
            else:
                continue
        t = binance_24h_ticker(b)
        if not t: 
            continue
        quote_vol = float(t.get("quoteVolume", 0.0))  # USDT volume
        if quote_vol < 8_000_000:
            continue
        bid, ask = binance_best_bid_ask(b)
        if not bid or not ask or ask == 0:
            continue
        spread_pct = (ask - bid) / ask * 100.0
        if spread_pct > 0.5:
            continue
        rows.append({
            "rev_symbol": sym,
            "binance_symbol": b,
            "bid": bid, "ask": ask, "spread_pct": spread_pct,
            "last_price": float(t.get("lastPrice", 0.0)),
            "quote_vol": quote_vol
        })
        time.sleep(0.05)  # polite
    return pd.DataFrame(rows)

# ----------- Signal detection ----------
def binance_klines_1m(symbol, limit=60):
    return binance_klines(symbol, "1m", limit)

def binance_trades(symbol, limit=500):
    js = safe_get(f"{BINANCE}/api/v3/aggTrades", {"symbol":symbol, "limit":limit})
    return js or []

def atr_1m(kl, period=14):
    # True range with 1m klines
    if len(kl) < period + 1:
        return None
    highs = np.array([float(k[2]) for k in kl], dtype=float)
    lows  = np.array([float(k[3]) for k in kl], dtype=float)
    closes= np.array([float(k[4]) for k in kl], dtype=float)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = pd.Series(tr).rolling(period).mean().iloc[-1]
    return float(atr)

def rvol_last_minute_vs_15m_median(kl):
    # RVOL = last 1m volume / median( last 15m volumes )
    if len(kl) < 16:
        return None
    vols = [float(k[5]) for k in kl[-16:]]  # 16 includes the last minute
    last_v = vols[-1]
    med15 = np.median(vols[:-1])
    if med15 == 0: 
        return None
    return float(last_v / med15)

def last_15m_high(kl):
    if len(kl) < 15: return None
    highs = [float(k[2]) for k in kl[-15:]]
    return float(np.max(highs))

def micro_pullback_conf(symbol, breakout_high):
    """
    Approximate 15–45 sec pullback ≤ 0.6% from high using recent aggTrades.
    We look at ~ last 60 seconds of trades:
      - price dipped ≤ 0.6% from breakout high
      - then price stabilizes (median of last 10 trades within that dip range)
    We cannot truly read bid absorption w/out L2; this is a proxy.
    """
    trades = binance_trades(symbol, limit=400)  # ~5-20s coverage depending on activity
    if not trades:
        return False, None, None
    # Each trade: p=price, q=qty, T=ms
    prices = [float(t["p"]) for t in trades]
    if len(prices) < 20:
        return False, None, None
    hi = breakout_high
    max_dip = 0.006 * hi   # 0.6%
    low_allowed = hi - max_dip
    min_after = min(prices[-200:])  # recent window
    if min_after < low_allowed - 1e-9:
        # dipped too much
        return False, None, None
    # define pullback low and a simple "hold" check (stabilization)
    pb_low = min(prices[-100:])
    hold = np.median(prices[-10:]) >= pb_low * 0.999  # stable
    return bool(hold), float(pb_low), float(np.median(prices[-5:]))

def select_signals(universe_df, regime_strong):
    """
    For each eligible symbol:
      - 1m price +1.8% to +4% vs prev close
      - RVOL >= 4.0
      - break above last 15m high with follow-through
      - micro pullback check via aggTrades proxy
    Return list of candidates with scores (RVOL, volume, spread, etc.)
    """
    candidates = []
    for _, row in universe_df.iterrows():
        sym = row["binance_symbol"]
        rev = row["rev_symbol"]
        kl = binance_klines_1m(sym, 60)
        if len(kl) < 20:
            continue
        prev_close = float(kl[-2][4])
        last_close = float(kl[-1][4])
        pct = (last_close - prev_close) / prev_close * 100.0
        if not (1.8 <= pct <= 4.0):
            continue
        rvol = rvol_last_minute_vs_15m_median(kl)
        if rvol is None or rvol < 4.0:
            continue
        h15 = last_15m_high(kl)
        if h15 is None or last_close <= h15:
            continue
        # "follow-through" -> last close > h15 by a small margin
        if (last_close - h15) / h15 < 0.001:  # >0.1%
            continue
        # micro pullback within last ~minute
        ok, pb_low, near_mid = micro_pullback_conf(sym, breakout_high=last_close)
        if not ok or pb_low is None:
            continue

        # ATR(1m) for targets
        a = atr_1m(kl, 14)
        if not a or a <= 0:
            continue

        entry = float(near_mid or last_close)
        t1 = entry + 0.8 * a
        t2 = entry + 1.5 * a
        stop = float(pb_low)  # invalidation = lose pullback low

        score = (rvol * 1.0) + (min(50.0, row["quote_vol"]/1e6) * 0.05) - (row["spread_pct"]*0.2)
        candidates.append({
            "rev_symbol": rev,
            "binance_symbol": sym,
            "entry": entry,
            "t1": t1, "t2": t2,
            "stop": stop,
            "rvol": rvol,
            "spread_pct": float(row["spread_pct"]),
            "quote_vol": float(row["quote_vol"]),
            "last_price": float(row["last_price"]),
            "score": float(score)
        })
        time.sleep(0.05)
    # sort by score desc
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

# ----------- Position/risk ----------
def risk_allocation(regime_strong):
    max_alloc = 0.60 if regime_strong else 0.30
    per_trade_risk = 0.012 * EQUITY_USD  # 1.2% equity per trade
    return max_alloc, per_trade_risk

def equity_preservation_breach():
    # Floor + buffer check
    if EQUITY_USD - FLOOR_USD < FLOOR_BUFFER:
        return True, "Equity buffer below required buffer."
    # Daily loss cap -3%
    if DAILY_PNL_PCT <= -3.0:
        return True, "Daily loss cap reached."
    return False, ""

def compute_size(entry, stop, regime_strong):
    max_alloc, per_trade_risk_usd = risk_allocation(regime_strong)
    # risk per unit:
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    units = per_trade_risk_usd / risk_per_unit
    notional = units * entry
    # cap by max allocation of equity
    cap = EQUITY_USD * max_alloc
    if notional > cap:
        notional = cap
        units = notional / entry
    # cap by available CASH
    if notional > CASH_USD:
        notional = CASH_USD
        units = notional / entry
    return float(max(0.0, notional))

# ----------- Portfolio rotation stub ----------
# We don't have live portfolio holdings from Revolut via API, so we leave a placeholder.
def weakest_holding_to_sell():
    # In your Action you can set PORTFOLIO_JSON env with current non-ETH/DOT holdings
    try:
        pf = os.environ.get("PORTFOLIO_JSON","")
        if not pf: 
            return "Weakest non-ETH/DOT holding on support break"
        j = json.loads(pf)
        # choose smallest 7d perf or lowest RS; here we pick min market value
        non_protected = [x for x in j if x.get("symbol","").upper() not in ("ETH","DOT")]
        if not non_protected:
            return "None (no non-ETH/DOT holdings)"
        # pick lowest market value
        s = sorted(non_protected, key=lambda x: float(x.get("value_usd",0)))[0]
        return f"{s.get('name','Unknown')} ({s.get('symbol','?')})"
    except Exception:
        return "Weakest non-ETH/DOT holding on support break"

# ----------- Main run ----------
def main():
    out = {"time": now_utc().isoformat()}
    # 0) Preservation checks
    breach, reason = equity_preservation_breach()
    out["equity"] = EQUITY_USD
    out["cash"] = CASH_USD
    out["breach"] = breach
    out["breach_reason"] = reason

    # 1) Regime
    reg = btc_regime()
    out["regime"] = reg

    if breach:
        # A) Preservation breach
        txt = f"Exit weakest: {weakest_holding_to_sell()} and raise cash immediately. Floor ${FLOOR_USD} with ${FLOOR_BUFFER} buffer breached or daily loss cap."
        decision = {
            "type":"A",
            "text": f"Raise cash now — {reason}. Exit weakest non-ETH, non-DOT holding immediately and halt new entries."
        }
        with open(SIG_PATH, "w") as f: json.dump(decision, f, indent=2)
        with open(SNAP_PATH,"w") as f: json.dump(out, f, indent=2)
        print("→ Output immediate rotation/raise-cash instructions.")
        print("Exit weakest non-ETH, non-DOT holding now and raise cash. Halt new trades.")
        return

    strong = reg.get("strong", False) if reg.get("ok") else False

    # 2) Universe
    df_map = read_mapping()
    uni = eligible_universe(df_map)
    out["universe_count"] = int(len(uni))

    # 3) Signals
    cands = select_signals(uni, regime_strong=strong)
    out["candidates"] = cands

    # 4) Choose best one (and size)
    if len(cands) == 0:
        with open(SIG_PATH, "w") as f: json.dump({"type":"C","text":"Hold and wait."}, f, indent=2)
        with open(SNAP_PATH,"w") as f: json.dump(out, f, indent=2)
        print("Hold and wait.")
        return

    # If regime weak -> take only top 1 (we always output one anyway)
    best = cands[0]
    buy_amount = compute_size(best["entry"], best["stop"], regime_strong=strong)

    # If buffer would fall below floor+buffer after this buy, trim buy to keep >= floor
    floor_need = FLOOR_USD + FLOOR_BUFFER
    post_equity_buffer = EQUITY_USD - floor_need  # equity is notional; we assume buy uses CASH only
    if post_equity_buffer < 0:
        # Already under planned buffer; do not buy
        with open(SIG_PATH, "w") as f: json.dump({"type":"C","text":"Hold and wait."}, f, indent=2)
        with open(SNAP_PATH,"w") as f: json.dump(out, f, indent=2)
        print("Hold and wait.")
        return

    # Ensure we don't reduce cash below minimal buffer need
    if CASH_USD - buy_amount < 0:
        buy_amount = max(0.0, CASH_USD)

    # Compose final B output
    target_str = f"T1 {best['t1']:.6f}, T2 {best['t2']:.6f}"
    stop_plan = f"Invalidation: lose pullback low {best['stop']:.6f}; trail after +1.0R; exit if stall > 5 min."
    sell_what = weakest_holding_to_sell()

    decision_text = [
        f"• {best['rev_symbol']} + {best['binance_symbol']}",
        f"• Entry price {best['entry']:.6f}",
        f"• Target {target_str}",
        f"• Stop-loss / exit plan {stop_plan}",
        f"• What to sell from portfolio (excluding ETH, DOT) {sell_what}",
        f"• Exact USD buy amount so total equity ≥ ${FLOOR_USD:.0f} {buy_amount:.2f}"
    ]
    final_txt = "\n".join(decision_text)

    with open(SIG_PATH, "w") as f: json.dump({"type":"B","candidate":best,"buy_usd":buy_amount,"text_lines":decision_text}, f, indent=2)
    with open(SNAP_PATH,"w") as f: json.dump(out, f, indent=2)

    # Print EXACTLY the requested B format:
    print(f"• {best['rev_symbol']} + {best['binance_symbol']}")
    print(f"• Entry price {best['entry']:.6f}")
    print(f"• Target {target_str}")
    print(f"• Stop-loss / exit plan {stop_plan}")
    print(f"• What to sell from portfolio (excluding ETH, DOT) {sell_what}")
    print(f"• Exact USD buy amount so total equity ≥ ${FLOOR_USD:.0f} {buy_amount:.2f}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        main()
    except Exception as e:
        # Fail-safe: never print extra text; default to "Hold and wait."
        with open(SIG_PATH, "w") as f: json.dump({"type":"C","error":str(e)}, f, indent=2)
        print("Hold and wait.")
