# scripts/marketdata.py
# Lightweight market data helpers for analyses.py
#
# Exposes:
#   - get_btc_5m_klines(limit=200) -> pd.DataFrame[time, open, high, low, close, volume]
#   - ema(series: pd.Series, span: int) -> pd.Series
#   - vwap(df: pd.DataFrame, price_col='close', vol_col='volume', window=None) -> pd.Series
#
# APIs used (in order with graceful fallback):
#   1) Binance    GET /api/v3/klines?symbol=BTCUSDT&interval=5m&limit=...
#   2) Coinbase   GET /products/BTC-USD/candles?granularity=300
#   3) Kraken     GET /0/public/OHLC?pair=XBTUSD&interval=5
#   4) Bitstamp   GET /api/v2/ohlc/btcusd/?step=300&limit=...

from __future__ import annotations

import time
from typing import Optional, Dict, List, Tuple, Any
import requests
import pandas as pd
import math
import json
from datetime import datetime, timezone



# ---------- utilities ----------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "rev-crypto-mapper/1.0 (+github actions)",
            "Accept": "application/json",
        }
    )
    s.timeout = 15
    return s


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize dtypes & order
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    """Plain EMA identical to TradingView's default (no adjust)."""
    return series.ewm(span=span, adjust=False).mean()


def vwap(
    df: pd.DataFrame,
    price_col: str = "close",
    vol_col: str = "volume",
    window: Optional[int] = None,
) -> pd.Series:
    """
    VWAP over the whole session (default) or a rolling window (in rows).
    """
    price = pd.to_numeric(df[price_col], errors="coerce")
    vol = pd.to_numeric(df[vol_col], errors="coerce")
    pv = price * vol
    if window is None:
        return pv.cumsum() / vol.cumsum()
    # rolling, min_periods to avoid NaNs at start
    roll_pv = pv.rolling(window=window, min_periods=1).sum()
    roll_v = vol.rolling(window=window, min_periods=1).sum()
    return roll_pv / roll_v


# ---------- API adapters ----------

def _binance_btc_5m(limit: int, s: requests.Session) -> Optional[pd.DataFrame]:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": int(limit)}
    r = s.get(url, params=params)
    if r.status_code != 200:
        return None
    raw: List[List] = r.json()
    if not raw:
        return None
    # Binance kline payload indices
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    out = df[["time", "open", "high", "low", "close", "volume"]]
    return _ensure_df(out)


def _coinbase_btc_5m(limit: int, s: requests.Session) -> Optional[pd.DataFrame]:
    # Coinbase returns in reverse chronological order
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": 300}  # 5 minutes
    r = s.get(url, params=params)
    if r.status_code != 200:
        return None
    raw = r.json()
    if not isinstance(raw, list) or not raw:
        return None
    # Each row: [ time, low, high, open, close, volume ]
    df = pd.DataFrame(raw, columns=["ts", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    out = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time")
    # respect limit from the tail
    if len(out) > limit:
        out = out.iloc[-limit:]
    return _ensure_df(out)


def _kraken_btc_5m(limit: int, s: requests.Session) -> Optional[pd.DataFrame]:
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 5}
    r = s.get(url, params=params)
    if r.status_code != 200:
        return None
    j = r.json()
    if "result" not in j:
        return None
    # Kraken nests by pair key (XBTUSD or XXBTZUSD depending on venue)
    result = next(iter(j["result"].values()))
    if not isinstance(result, list) or not result:
        return None
    # Row: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(result, columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    out = df[["time", "open", "high", "low", "close", "volume"]]
    if len(out) > limit:
        out = out.iloc[-limit:]
    return _ensure_df(out)


def _bitstamp_btc_5m(limit: int, s: requests.Session) -> Optional[pd.DataFrame]:
    url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
    params = {"step": 300, "limit": max(100, int(limit))}
    r = s.get(url, params=params)
    if r.status_code != 200:
        return None
    j = r.json()
    data = j.get("data", {}).get("ohlc", [])
    if not data:
        return None
    df = pd.DataFrame(data)
    # Bitstamp strings
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    out = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time")
    if len(out) > limit:
        out = out.iloc[-limit:]
    return _ensure_df(out)


# ---------- public facade ----------

def get_btc_5m_klines(limit: int = 200) -> pd.DataFrame:
    """
    Return a 5-minute BTC/USD (or USDT) OHLCV DataFrame with columns:
    time (UTC), open, high, low, close, volume
    Uses robust exchange fallbacks.
    """
    s = _session()

    # Try Binance
    try:
        df = _binance_btc_5m(limit, s)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Try Coinbase
    try:
        df = _coinbase_btc_5m(limit, s)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Try Kraken
    try:
        df = _kraken_btc_5m(limit, s)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Try Bitstamp
    try:
        df = _bitstamp_btc_5m(limit, s)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # If everything failed, return an empty, well-formed frame
    return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
