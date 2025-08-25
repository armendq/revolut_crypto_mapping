# scripts/analyses.py
# Heavy scanner for Revolut-listed crypto: multi-timeframe breakouts, 8h momentum,
# robust HTTP retries, and multi-source fallbacks. Produces:
# - public_runs/latest/summary.json
# - public_runs/latest/signals.json
# - public_runs/latest/market_snapshot.json
# - public_runs/latest/debug_scan.json
# - public_runs/latest/run_stats.json
#
# Run with:  python -m scripts.analyses

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ----------------------------------
# Configuration
# ----------------------------------

RUN_DIR = os.path.join("public_runs", "latest")
os.makedirs(RUN_DIR, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Revolut universe: try to read dynamically (preferred), else fall back to a curated list.
MAPPING_FILES = [
    os.path.join("data", "revolut_mapping.json"),
    os.path.join("data", "mapping.json"),
    os.path.join("data", "revolut_mapping.csv"),
]

# Conservative pacing to avoid API bans (Binance weight constraints).
REQ_SLEEP_SECONDS = 0.06  # ~16 req/s theoretical; we also batch work to be safer
MAX_RETRIES = 4
INITIAL_BACKOFF = 0.6

# Lookback windows
KLINE_LIMIT_1H = 500       # ~500 hours
KLINE_LIMIT_15M = 500      # ~125 hours
KLINE_LIMIT_4H = 200       # ~33 days

# Candidate selection thresholds
VOLUME_SURGE_MULT = 2.0
BREAKOUT_LOOKBACK_BARS = 96        # 4 days on 1h
SLOW_BURN_WINDOW_H = 8             # 8 hours window to catch ACS-type moves
SLOW_BURN_MIN_GAIN = 0.16          # +16% over the last 8 hours (tune up/down)
RSI_OB = 80                        # avoid ultra-overbought unless volume conf
RSI_LEN = 14
EMA_FAST = 20
EMA_SLOW = 200

# Regime check for BTC
BTC_SYMBOL = "BTCUSDT"

# DOT/ETH special handling
NON_ROTATE = {"ETH", "DOT"}  # they are not rotated; DOT is staked

# Optional CoinGecko mapping for tickers not on Binance (e.g., ACS)
COINGECKO_IDS = {
    "ACS": "access-protocol",
    "IMX": "immutable-x",
    "ARB": "arbitrum",
    "OP": "optimism",
    "ETH": "ethereum",
    "BTC": "bitcoin",
    "DOT": "polkadot",
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "TON": "the-open-network",
    "APT": "aptos",
    "SUI": "sui",
    # extend as needed
}

# ----------------------------------
# Utilities
# ----------------------------------

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[analyses] {ts} {msg}", flush=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_possible_mapping() -> List[str]:
    # Try JSON mapping files first
    for path in MAPPING_FILES:
        if os.path.exists(path) and path.endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Accept either {"ticker": "..."} list or {"revolut": [{"ticker":"..."}], ...}
                tickers = []
                if isinstance(data, list):
                    for row in data:
                        t = row.get("ticker") or row.get("symbol")
                        if t:
                            tickers.append(str(t).upper())
                elif isinstance(data, dict):
                    if "revolut" in data and isinstance(data["revolut"], list):
                        for row in data["revolut"]:
                            t = row.get("ticker") or row.get("symbol")
                            if t:
                                tickers.append(str(t).upper())
                    elif "tickers" in data and isinstance(data["tickers"], list):
                        tickers = [str(t).upper() for t in data["tickers"]]
                tickers = sorted(set([t for t in tickers if t.isalpha()]))
                if tickers:
                    log(f"loaded universe from {path} ({len(tickers)} tickers)")
                    return tickers
            except Exception as e:
                log(f"failed to read {path}: {e}")

    # Try CSV
    for path in MAPPING_FILES:
        if os.path.exists(path) and path.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                col = None
                for c in df.columns:
                    if c.lower() in {"ticker", "symbol"}:
                        col = c
                        break
                if col:
                    tickers = sorted(set([str(x).upper() for x in df[col].tolist() if isinstance(x, str)]))
                    if tickers:
                        log(f"loaded universe from {path} ({len(tickers)} tickers)")
                        return tickers
            except Exception as e:
                log(f"failed to read {path}: {e}")

    # Fallback curated list (make sure key assets + ACS are present)
    fallback = [
        "BTC","ETH","DOT","SOL","AVAX","MATIC","ADA","DOGE","TON","ARB","OP",
        "IMX","SUI","APT","ATOM","LINK","LTC","NEAR","ALGO","FIL","AAVE","APT",
        "INJ","RNDR","RUNE","FTM","XRP","HBAR","ETC","UNI","SAND","MANA","GALA",
        "APE","EGLD","KAS","TIA","JTO","PYTH","STRK","SEI","TIA","TRX","ROSE",
        "SATS","ORDI","FET","TAO","JUP","PEPE","WIF","BONK","ENA","AEVO",
        # small/mid caps
        "ACS","TIA","BOME","METIS","ARB","OP","IMX","ONDO","POLYX"
    ]
    tickers = sorted(set(fallback))
    log(f"loaded fallback universe ({len(tickers)} tickers)")
    return tickers


def with_retries(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.request(method, url, timeout=20, **kwargs)
            if resp.status_code == 200:
                return resp
            # 4xx/5xx -> retry with backoff
            log(f"GET fail: {url.split('?')[0]} http {resp.status_code} (try {attempt})")
        except requests.RequestException as e:
            log(f"GET error: {e} (try {attempt})")
        # jittered backoff
        time.sleep(backoff + random.random() * 0.5)
        backoff *= 1.8
    # Final request (return last response if present)
    # If we reached here, make one last attempt without sleeping
    try:
        resp = session.request(method, url, timeout=25, **kwargs)
        return resp
    except Exception as e:
        raise RuntimeError(f"HTTP failed after retries for {url}: {e}")


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, length: int = RSI_LEN) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ema_up = up.ewm(alpha=1/length, adjust=False).mean()
    ema_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    # True range
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def vwap(df: pd.DataFrame, length: int = 20) -> pd.Series:
    pv = df["close"] * df["volume"]
    return pv.rolling(length).sum() / df["volume"].rolling(length).sum()


def klines_to_df(rows: List[List]) -> pd.DataFrame:
    # Binance kline structure: [open_time, open, high, low, close, volume, close_time, qav, trades, tbav, tbqav, ignore]
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    data = []
    for r in rows:
        data.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[6])])
    df = pd.DataFrame(data, columns=cols)
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


# ----------------------------------
# Data fetchers
# ----------------------------------

class MarketData:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT})
        self.binance_hosts = [
            "https://api.binance.com",
            "https://data-api.binance.vision",
        ]
        self.host_idx = 0

    def _host(self) -> str:
        # rotate hosts
        host = self.binance_hosts[self.host_idx % len(self.binance_hosts)]
        self.host_idx += 1
        return host

    def fetch_binance_klines(self, symbol_usdt: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        # Try both hosts with retries
        params = {"symbol": symbol_usdt, "interval": interval, "limit": limit}
        for _ in range(len(self.binance_hosts)):
            base = self._host()
            url = f"{base}/api/v3/klines"
            resp = with_retries(self.s, "GET", url, params=params)
            if resp.status_code == 200:
                rows = resp.json()
                if not rows:
                    return None
                return klines_to_df(rows)
            # next host
        return None

    def fetch_binance_24hr(self, symbol_usdt: str) -> Optional[dict]:
        params = {"symbol": symbol_usdt}
        for _ in range(len(self.binance_hosts)):
            base = self._host()
            url = f"{base}/api/v3/ticker/24hr"
            resp = with_retries(self.s, "GET", url, params=params)
            if resp.status_code == 200:
                return resp.json()
            # try next host
        return None

    def fetch_coingecko_hourly(self, cg_id: str, days: int = 7) -> Optional[pd.DataFrame]:
        # CoinGecko market chart: hourly closes for up to 7 days
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
        resp = with_retries(self.s, "GET", url, params=params)
        if resp.status_code != 200:
            return None
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return None
        # Build hourly candles from close series (approximate OHLC using last/rolling)
        times = [pd.to_datetime(p[0], unit="ms", utc=True) for p in prices]
        closes = [float(p[1]) for p in prices]
        vols = [float(v[1]) for v in volumes] if volumes else [np.nan] * len(times)
        df = pd.DataFrame({"time": times, "close": closes, "volume": vols})
        # Derive OHLC as last-close proxy; compute high/low via rolling windows to avoid zero spreads
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df["close"].rolling(4, min_periods=1).max()
        df["low"] = df["close"].rolling(4, min_periods=1).min()
        return df[["time", "open", "high", "low", "close", "volume"]]


# ----------------------------------
# Signal logic
# ----------------------------------

@dataclass
class Candidate:
    ticker: str
    symbol: str               # e.g., "ACSUSDT" or "CG:access-protocol"
    source: str               # "binance" or "coingecko"
    price: float
    atr1h: float
    rsi1h: float
    ema20_1h: float
    ema200_1h: float
    vwap20_1h: float
    slowburn_8h_gain: float
    vol_surge_ratio: float
    breakout: bool
    breakout_bar_low: float
    momentum_score: float


def analyze_symbol(md: MarketData, ticker: str) -> Optional[Candidate]:
    """
    Fetch multi-timeframe data and evaluate candidate criteria.
    Preference: Binance (USDT pairs). Fallback: CoinGecko hourly.
    """
    symbol = f"{ticker.upper()}USDT"
    df_1h = md.fetch_binance_klines(symbol, "1h", KLINE_LIMIT_1H)
    source = "binance"
    if df_1h is None:
        # fallback to CoinGecko
        cg_id = COINGECKO_IDS.get(ticker.upper())
        if not cg_id:
            return None
        df_1h = md.fetch_coingecko_hourly(cg_id, days=7)
        if df_1h is None or len(df_1h) < 60:
            return None
        symbol = f"CG:{cg_id}"
        source = "coingecko"

    if len(df_1h) < 120:
        return None

    # indicators on 1h
    close = df_1h["close"]
    high = df_1h["high"]
    low = df_1h["low"]
    vol = df_1h["volume"].fillna(0)

    ema20 = compute_ema(close, EMA_FAST)
    ema200 = compute_ema(close, EMA_SLOW)
    rsi = compute_rsi(close, RSI_LEN)
    vwap20 = vwap(df_1h, 20)
    atr1h = atr(df_1h, 14)

    last = df_1h.iloc[-1]
    price = float(last["close"])
    last_ema20 = float(ema20.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_vwap = float(vwap20.iloc[-1])
    last_atr = float(atr1h.iloc[-1])

    # breakout setup (1h): price breaks above recent N-bar high with volume expansion
    lookback_high = float(high.tail(BREAKOUT_LOOKBACK_BARS).max())
    breakout = price > lookback_high * 1.001  # tiny buffer
    # breakout bar low approximation: last 3 bars low min
    breakout_bar_low = float(low.tail(3).min())

    # volume surge vs 20-bar average
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1] or 0)
    vol_surge_ratio = float(vol.iloc[-1] / max(vol_ma20, 1e-9)) if vol_ma20 > 0 else 0.0

    # 8h slow-burn gain (ACS-style)
    if len(close) >= 8:
        slow_burn_gain = price / float(close.iloc[-SLOW_BURN_WINDOW_H]) - 1.0
    else:
        slow_burn_gain = 0.0

    # Basic filters to be even considered:
    # - price above EMA20 and VWAP
    # - RSI not insanely OB unless volume is big
    base_pass = (price > last_ema20) and (price > last_vwap)
    rsi_ok = (last_rsi < RSI_OB) or (vol_surge_ratio >= VOLUME_SURGE_MULT * 1.5)

    if not base_pass or not rsi_ok:
        return None

    # Momentum score: blend of z-scores
    eps = 1e-9
    ema_gap = (price - last_ema20) / max(last_atr, price * 0.002 + eps)
    vwap_gap = (price - last_vwap) / max(last_atr, price * 0.002 + eps)
    trend_ok = 1.0 if price > last_ema200 else 0.0
    breakout_score = 1.0 if breakout else 0.0
    slow_burn_score = max(0.0, slow_burn_gain / max(SLOW_BURN_MIN_GAIN, eps))

    momentum = (
        0.8 * ema_gap +
        0.8 * vwap_gap +
        1.2 * breakout_score +
        0.8 * slow_burn_score +
        0.6 * trend_ok +
        0.3 * math.log(max(vol_surge_ratio, 1.0))
    )

    return Candidate(
        ticker=ticker.upper(),
        symbol=symbol,
        source=source,
        price=price,
        atr1h=last_atr,
        rsi1h=last_rsi,
        ema20_1h=last_ema20,
        ema200_1h=last_ema200,
        vwap20_1h=last_vwap,
        slowburn_8h_gain=slow_burn_gain,
        vol_surge_ratio=vol_surge_ratio,
        breakout=bool(breakout),
        breakout_bar_low=breakout_bar_low,
        momentum_score=float(momentum),
    )


def btc_regime(md: MarketData) -> Dict[str, object]:
    df = md.fetch_binance_klines(BTC_SYMBOL, "1h", 500)
    if df is None or len(df) < 220:
        return {"ok": False, "reason": "no_bars"}

    close = df["close"]
    ema20 = compute_ema(close, EMA_FAST)
    ema200 = compute_ema(close, EMA_SLOW)
    vw = vwap(df, 20)

    last = float(close.iloc[-1])
    ok = (last > float(ema20.iloc[-1])) and (last > float(ema200.iloc[-1])) and (last > float(vw.iloc[-1]))
    return {
        "ok": bool(ok),
        "price": last,
        "ema20": float(ema20.iloc[-1]),
        "ema200": float(ema200.iloc[-1]),
        "vwap20": float(vw.iloc[-1]),
    }


# ----------------------------------
# Main run
# ----------------------------------

def main() -> None:
    started = time.time()
    md = MarketData()

    tickers = read_possible_mapping()
    # Ensure we don't accidentally scan USD or stablecoins
    tickers = [t for t in tickers if t.upper() not in {"USDT", "USDC", "DAI", "EURS", "USDC.E"}]
    tickers = sorted(set(tickers))

    # Regime (BTC 1h)
    reg = btc_regime(md)
    log(f"regime: ok={reg.get('ok')} price={reg.get('price')} ema20={reg.get('ema20')} ema200={reg.get('ema200')} vwap20={reg.get('vwap20')}")

    # Scan universe
    debug_rows = []
    cands: List[Candidate] = []

    log(f"universe size: {len(tickers)}")
    for i, t in enumerate(tickers, start=1):
        try:
            cand = analyze_symbol(md, t)
            if cand:
                cands.append(cand)
                debug_rows.append({"ticker": t, **asdict(cand)})
            else:
                debug_rows.append({"ticker": t, "skip": True})
        except Exception as e:
            debug_rows.append({"ticker": t, "error": str(e)})
        # pacing
        time.sleep(REQ_SLEEP_SECONDS)
        if i % 25 == 0:
            log(f"scanned {i}/{len(tickers)}")

    # Rank candidates
    cands = sorted(cands, key=lambda c: c.momentum_score, reverse=True)

    # Build summary + signals
    now_utc = datetime.now(timezone.utc).isoformat()

    summary: Dict[str, object] = {
        "status": "ok",
        "generated_at": now_utc,
        "regime": reg,
        "counts": {
            "universe": len(tickers),
            "candidates": len(cands),
        },
        "notes": [
            "Breakout = price > recent high (1h), vol surge & EMA/VWAP alignment.",
            f"Slow burn = {SLOW_BURN_WINDOW_H}h gain ≥ {int(SLOW_BURN_MIN_GAIN*100)}%.",
            "DOT is staked & untouchable; ETH/DOT not rotated.",
        ],
    }

    # Decide signal:
    signals = {"type": "C", "why": "no breakout", "ticker": None}
    if cands:
        top = cands[0]
        # We treat a fresh breakout as a 'B' (breakout) signal; otherwise 'C' (candidates only).
        if top.breakout and top.vol_surge_ratio >= VOLUME_SURGE_MULT:
            signals = {
                "type": "B",
                "ticker": top.ticker,
                "entry": round(top.price, 8),
                "stop": round(top.breakout_bar_low, 8),
                "atr": round(top.atr1h, 8),
                "t1_mul_atr": 0.8,
                "t2_mul_atr": 1.5,
                "trail_after_r": 1.0,
                "source": top.source,
                "symbol": top.symbol,
                "comment": "Breakout with volume; use breakout bar low as stop. Trail after +1R.",
            }
        else:
            signals = {
                "type": "C",
                "why": "no confirmed breakout; listing top momentum candidates",
            }

    # Prepare candidates view for summary.json
    summary["candidates"] = [
        {
            "ticker": c.ticker,
            "source": c.source,
            "symbol": c.symbol,
            "price": round(c.price, 8),
            "atr1h": round(c.atr1h, 8),
            "rsi1h": round(c.rsi1h, 2),
            "ema20_1h": round(c.ema20_1h, 8),
            "ema200_1h": round(c.ema200_1h, 8),
            "vwap20_1h": round(c.vwap20_1h, 8),
            "slowburn_8h_gain": round(c.slowburn_8h_gain, 4),
            "vol_surge_ratio": round(c.vol_surge_ratio, 3),
            "breakout": c.breakout,
            "breakout_bar_low": round(c.breakout_bar_low, 8),
            "momentum_score": round(c.momentum_score, 4),
        }
        for c in cands[:12]
    ]

    # Respect ETH/DOT not-rotated note (informational; actual sizing rules applied at consumer step)
    for c in summary["candidates"]:
        if c["ticker"] in NON_ROTATE:
            c["note"] = "Not rotated (hold/core)."

    # Write files
    save_json(os.path.join(RUN_DIR, "summary.json"), {
        **summary,
        "signals": signals,
    })

    save_json(os.path.join(RUN_DIR, "signals.json"), signals)

    # Market snapshot (BTC only for now)
    save_json(os.path.join(RUN_DIR, "market_snapshot.json"), {
        "generated_at": now_utc,
        "btc": reg,
    })

    # Debug scan
    save_json(os.path.join(RUN_DIR, "debug_scan.json"), debug_rows)

    # Run stats
    elapsed = time.time() - started
    save_json(os.path.join(RUN_DIR, "run_stats.json"), {
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed, 2),
        "scanned": len(tickers),
        "candidates": len(cands),
        "source": "analyses.py/heavy",
    })

    log(f"wrote summary with {len(cands)} candidates in {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
            # Make sure CI still emits files (so downstream steps don't choke)
            err = {"status": "error", "message": str(e), "time": datetime.now(timezone.utc).isoformat()}
            save_json(os.path.join(RUN_DIR, "summary.json"), err)
            save_json(os.path.join(RUN_DIR, "run_stats.json"), {"error": str(e)})
            log(f"FATAL: {e}")
            sys.exit(1)