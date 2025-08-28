#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline – robust scanner for pre-breakouts and breakouts.

Key features:
- Uses data-api.binance.vision (works in restricted regions)
- Never calls Binance without a symbol (avoids 451/403)
- Multi-timeframe trend filters (4h + 1h), momentum confirm (15m/5m)
- Pre-breakout detection: proximity to HH20 with rising ATR + volume thrust
- Candidate ranking: volume z-score, proximity/ATR, RS vs BTC
- Always emits a valid summary.json under public_runs/latest/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ----------------------------- CONFIG ---------------------------------

TZ_NAME = "Europe/Prague"
TZ = ZoneInfo(TZ_NAME) if ZoneInfo else timezone.utc

BASE_URL = "https://data-api.binance.vision"

# Intervals & limits
INTERVALS = {
    "4h": ("4h", 500),
    "1h": ("1h", 600),
    "15m": ("15m", 400),
    "5m": ("5m", 400),
}

# ATR / EMA parameters
ATR_LEN = 14
EMA_FAST = 20
EMA_SLOW = 200
SWING_LOOKBACK = 20  # HH20 / LL20

# Pre-breakout heuristics
PROX_ATR_MIN = 0.05   # close at least this many ATRs below HH20
PROX_ATR_MAX = 0.35   # within 0.35 ATR of HH20
VOL_Z_MIN_PRE = 1.2   # volume z-score for pre-breakout
VOL_Z_MIN_BREAK = 1.8 # stronger thrust for confirmed breakout
BREAK_BUFFER_ATR = 0.10  # entry buffer above HH20 for B signals

# Ranking
MAX_CANDIDATES = 10

# Revolut universe (trimmed to non-stables later)
REVOLUT_TICKERS: Dict[str, str] = {
    "1INCH": "1INCH", "AAVE": "Aave", "ACA": "Acala Token", "ADA": "Cardano", "AERGO": "Aergo",
    "AGLD": "Adventure Gold", "AIOZ": "AIOZ Network", "AKT": "Akash Network", "ALGO": "Algorand",
    "ALICE": "MyNeighborAlice", "ALPHA": "Alpha Venture DAO", "AMP": "AMP", "ANKR": "Ankr",
    "APE": "ApeCoin", "API3": "API3", "APT": "Aptos", "AR": "Arweave", "ARB": "Arbitrum",
    "ARKM": "Arkham", "ARPA": "ARPA", "ASTR": "Astar", "ATOM": "Cosmos", "AUDIO": "Audius",
    "AVAX": "Avalanche", "AXS": "Axie Infinity", "BAL": "Balancer", "BAND": "Band Protocol",
    "BAT": "Basic Attention Token", "BCH": "Bitcoin Cash", "BICO": "Biconomy", "BLUR": "Blur",
    "BLZ": "Bluzelle", "BNC": "Bifrost", "BNT": "Bancor", "BONK": "Bonk", "BOND": "BarnBridge",
    "BTC": "Bitcoin", "BTG": "Bitcoin Gold", "BTRST": "Braintrust", "BTT": "BitTorrent",
    "CELO": "Celo", "CELR": "Celer Network", "CFX": "Conflux", "CHZ": "Chiliz", "CHR": "Chromia",
    "CLV": "Clover Finance", "COMP": "Compound", "COTI": "COTI", "CQT": "Covalent",
    "CRO": "Cronos", "CRV": "Curve DAO Token", "CTSI": "Cartesi", "CVC": "Civic",
    "CVX": "Convex Finance", "DAI": "Dai", "DAR": "Mines of Dalarnia", "DASH": "Dash",
    "DENT": "Dent", "DGB": "DigiByte", "DIMO": "DIMO", "DNT": "district0x", "DOGE": "Dogecoin",
    "DOT": "Polkadot", "DREP": "DREP", "DYDX": "dYdX", "EGLD": "MultiversX", "ENA": "Ethena",
    "ENJ": "Enjin Coin", "ENS": "Ethereum Name Service", "EOS": "EOS", "ETC": "Ethereum Classic",
    "ETH": "Ethereum", "ETHW": "EthereumPoW", "EUL": "Euler", "FET": "Fetch.ai", "FIDA": "Bonfida",
    "FIL": "Filecoin", "FLOKI": "FLOKI", "FLOW": "Flow", "FLR": "Flare", "FORTH": "Ampleforth Governance Token",
    "FTM": "Fantom", "FXS": "Frax Share", "GALA": "Gala", "GFI": "Goldfinch", "GHST": "Aavegotchi",
    "GLM": "Golem", "GLMR": "Moonbeam", "GMT": "STEPN", "GMX": "GMX", "GNO": "Gnosis",
    "GNS": "Gains Network", "GODS": "Gods Unchained", "GRT": "The Graph", "HBAR": "Hedera",
    "HFT": "Hashflow", "HIGH": "Highstreet", "HOPR": "HOPR", "HOT": "Holo", "ICP": "Internet Computer",
    "IDEX": "IDEX", "ILV": "Illuvium", "IMX": "Immutable", "INJ": "Injective", "IOTX": "IoTeX",
    "JASMY": "JasmyCoin", "JTO": "Jito", "JUP": "Jupiter", "KAVA": "Kava", "KDA": "Kadena",
    "KNC": "Kyber Network", "KSM": "Kusama", "LDO": "Lido DAO", "LINK": "Chainlink",
    "LPT": "Livepeer", "LQTY": "Liquity", "LRC": "Loopring", "LSK": "Lisk", "LTC": "Litecoin",
    "MAGIC": "MAGIC", "MANA": "Decentraland", "MASK": "Mask Network", "MATIC": "Polygon",
    "MC": "Merit Circle", "MDT": "Measurable Data Token", "MINA": "Mina", "MKR": "Maker",
    "MLN": "Enzyme", "MPL": "Maple", "MTRG": "Meter Governance", "NEAR": "NEAR Protocol",
    "NEXO": "Nexo", "NKN": "NKN", "OCEAN": "Ocean Protocol", "OGN": "Origin Protocol",
    "OMG": "OMG Network", "ONDO": "Ondo", "ONG": "Ontology Gas", "ONT": "Ontology",
    "OP": "Optimism", "ORCA": "Orca", "OXT": "Orchid", "PAXG": "PAX Gold", "PENDLE": "Pendle",
    "PEPE": "PEPE", "PERP": "Perpetual Protocol", "PIXEL": "Pixels", "PLA": "PlayDapp",
    "POKT": "Pocket Network", "POLS": "Polkastarter", "POLYX": "Polymesh", "POND": "Marlin",
    "POWR": "Powerledger", "PUNDIX": "Pundi X", "QI": "BENQI", "QNT": "Quant", "QTUM": "Qtum",
    "RAD": "Radicle", "RARE": "SuperRare", "RARI": "Rarible", "RAY": "Raydium", "REN": "Ren",
    "REQ": "Request", "RLC": "iExec RLC", "RLY": "Rally", "RNDR": "Render", "RON": "Ronin",
    "ROSE": "Oasis Network", "RPL": "Rocket Pool", "RSR": "Reserve Rights", "SAND": "The Sandbox",
    "SC": "Siacoin", "SGB": "Songbird", "SHIB": "Shiba Inu", "SKL": "SKALE Network",
    "SLP": "Smooth Love Potion", "SNT": "Status", "SNX": "Synthetix", "SOL": "Solana",
    "SPELL": "Spell Token", "STG": "Stargate Finance", "STORJ": "Storj", "STRK": "Starknet",
    "STX": "Stacks", "SUI": "Sui", "SUSHI": "SushiSwap", "SYN": "Synapse", "SYNTH": "SYNTH",
    "T": "Threshold", "TEL": "Telcoin", "TIA": "Celestia", "TON": "Toncoin", "TRB": "Tellor",
    "TRU": "TrueFi", "TRX": "TRON", "UMA": "UMA", "UNI": "Uniswap", "USDC": "USD Coin",
    "USDT": "Tether", "VET": "VeChain", "VOXEL": "Voxies", "VTHO": "VeThor Token", "WAVES": "Waves",
    "WAXP": "WAX", "WBTC": "Wrapped Bitcoin", "WCFG": "Wrapped Centrifuge", "WLD": "Worldcoin",
    "WOO": "WOO Network", "XLM": "Stellar", "XMR": "Monero", "XRP": "XRP", "XTZ": "Tezos",
    "XVS": "Venus", "YFI": "yearn.finance", "YGG": "Yield Guild Games", "ZEC": "Zcash",
    "ZETA": "ZetaChain", "ZIL": "Zilliqa", "ZRX": "0x"
}

# Exclude stablecoins from scanning
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}

# Rotation-exempt info (respected downstream; we annotate for clarity)
ROTATION_EXEMPT = {"ETH", "DOT"}

# -------------------------- HELPERS -----------------------------------

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[analyses {ts}] {msg}", flush=True)

def human_time(ts: Optional[datetime] = None) -> str:
    dt = ts or datetime.now(timezone.utc).astimezone(TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def safe_get(url: str, params: Optional[dict] = None, retries: int = 3, timeout: int = 20, sleep_s: float = 0.6) -> Optional[requests.Response]:
    """GET with retries/backoff. Returns Response or None."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            # transient/rate-limit statuses
            if r.status_code in (418, 429, 451, 403, 504, 500):
                time.sleep(sleep_s * attempt)
            else:
                time.sleep(sleep_s * 0.5)
        except requests.RequestException:
            time.sleep(sleep_s * attempt)
    return None

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    alpha = 2.0 / (period + 1.0)
    out: List[float] = []
    prev = series[0]
    out.append(prev)
    for x in series[1:]:
        prev = alpha * x + (1 - alpha) * prev
        out.append(prev)
    return out

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = ATR_LEN) -> List[float]:
    if len(highs) < 2:
        return [0.0] * len(highs)
    trs = [0.0]
    for i in range(1, len(highs)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return ema(trs, period)

def swing_high(series: List[float], lookback: int) -> List[float]:
    out: List[float] = []
    for i in range(len(series)):
        if i < lookback:
            out.append(max(series[: i + 1]))
        else:
            out.append(max(series[i - lookback + 1 : i + 1]))
    return out

def swing_low(series: List[float], lookback: int) -> List[float]:
    out: List[float] = []
    for i in range(len(series)):
        if i < lookback:
            out.append(min(series[: i + 1]))
        else:
            out.append(min(series[i - lookback + 1 : i + 1]))
    return out

def pct_change(series: List[float], n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    a, b = series[-n - 1], series[-1]
    if a == 0:
        return 0.0
    return (b - a) / a

def vol_zscore(vols: List[float], window: int = 50) -> float:
    if len(vols) < window + 1:
        return 0.0
    sample = vols[-(window + 1) : -1]
    mu = statistics.fmean(sample)
    sd = statistics.pstdev(sample)
    if sd == 0:
        return 0.0
    return (vols[-1] - mu) / sd

def fetch_exchange_usdt_symbols() -> Dict[str, dict]:
    """Return dict of USDT spot symbols metadata keyed by symbol (e.g., 'LPTUSDT')."""
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    r = safe_get(url, retries=3, timeout=25)
    if not r:
        return {}
    data = r.json()
    out: Dict[str, dict] = {}
    for s in data.get("symbols", []):
        if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT" and s.get("isSpotTradingAllowed", True):
            out[s["symbol"]] = s
    return out

def fetch_klines(symbol: str, interval: str, limit: int) -> Optional[List[List[Any]]]:
    """Fetch raw klines array for a given symbol/interval."""
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = safe_get(url, params=params, retries=3, timeout=20)
    if not r:
        return None
    try:
        data = r.json()
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None

def parse_klines(raw: List[List[Any]]) -> Dict[str, List[float]]:
    """Return dict with open_time(ms), open, high, low, close, volume as lists (floats)."""
    ot: List[int] = []
    o: List[float] = []
    h: List[float] = []
    l: List[float] = []
    c: List[float] = []
    v: List[float] = []
    for row in raw:
        # [ openTime, open, high, low, close, volume, closeTime, ... ]
        ot.append(int(row[0]))
        o.append(float(row[1]))
        h.append(float(row[2]))
        l.append(float(row[3]))
        c.append(float(row[4]))
        v.append(float(row[5]))
    return {"open_time": ot, "open": o, "high": h, "low": l, "close": c, "volume": v}

# -------------------------- CORE LOGIC --------------------------------

def analyze_symbol(symbol: str, btc_1h_close: List[float]) -> Optional[dict]:
    """Analyze single symbol; return candidate dict or confirmed breakout dict with 'signal' key."""
    # Fetch required timeframes
    req: Dict[str, Dict[str, List[float]]] = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw = fetch_klines(symbol, interval, limit)
        if not raw or len(raw) < max(EMA_SLOW + 5, SWING_LOOKBACK + 5):
            return None
        req[tf] = parse_klines(raw)

    one = req["1h"]
    four = req["4h"]
    fifteen = req["15m"]
    five = req["5m"]

    close_1h = one["close"]
    high_1h = one["high"]
    low_1h = one["low"]
    vol_1h = one["volume"]

    # Trend filters
    ema20_1h = ema(close_1h, EMA_FAST)
    ema200_1h = ema(close_1h, EMA_SLOW)
    ema20_4h = ema(four["close"], EMA_FAST)
    ema200_4h = ema(four["close"], EMA_SLOW)

    if len(ema200_1h) < 5 or len(ema200_4h) < 5:
        return None

    ema20_slope_1h = ema20_1h[-1] - ema20_1h[-4]
    ema20_slope_4h = ema20_4h[-1] - ema20_4h[-4]

    trend_ok = (
        ema20_slope_1h > 0
        and ema20_slope_4h > 0
        and close_1h[-1] > ema200_1h[-1]
        and four["close"][-1] > ema200_4h[-1]
    )

    # ATR & swings on 1h
    atr_1h = atr(high_1h, low_1h, close_1h, ATR_LEN)
    if atr_1h[-1] <= 0:
        return None

    hh20 = swing_high(high_1h, SWING_LOOKBACK)
    ll20 = swing_low(low_1h, SWING_LOOKBACK)

    # Volume context
    vz = vol_zscore(vol_1h, window=50)

    # Pre-breakout proximity (to HH20)
    prox = (hh20[-1] - close_1h[-1]) / max(1e-9, atr_1h[-1])  # in ATRs
    atr_rising = atr_1h[-1] > atr_1h[-2] > atr_1h[-3]

    # Momentum confirms (lower TFs): price > EMA20 and EMA20 sloping up
    def mom_ok(tf_data: Dict[str, List[float]]) -> bool:
        c = tf_data["close"]
        e20 = ema(c, EMA_FAST)
        if len(e20) < 5:
            return False
        return (c[-1] > e20[-1]) and (e20[-1] > e20[-3])

    mom15 = mom_ok(fifteen)
    mom5 = mom_ok(five)
    lower_tf_ok = mom15 and mom5

    # RS vs BTC (1h)
    rs_strength = 0.0
    if btc_1h_close and len(btc_1h_close) == len(close_1h):
        rs_series = [c / b if b != 0 else 0.0 for c, b in zip(close_1h, btc_1h_close)]
        rs_strength = pct_change(rs_series, n=10)

    last_close = close_1h[-1]
    last_low = low_1h[-1]
    last_atr = atr_1h[-1]

    reasons: List[str] = []
    if trend_ok:
        reasons.append("TrendOK(1h&4h)")
    if atr_rising:
        reasons.append("ATR up")
    if vz > 0:
        reasons.append(f"VolZ={vz:.2f}")
    if lower_tf_ok:
        reasons.append("LowerTF OK")

    score = 0.0
    if vz > 0:
        score += vz
    if prox > 0:
        score += 1.0 / (min(max(prox, 0.01), 2.0))
    score += max(0.0, 5.0 * rs_strength)
    if lower_tf_ok:
        score += 0.5

    breakout_level = hh20[-1]
    confirmed_breakout = (
        trend_ok
        and last_close > (breakout_level + BREAK_BUFFER_ATR * last_atr)
        and vz >= VOL_Z_MIN_BREAK
        and lower_tf_ok
    )

    pre_breakout = (
        trend_ok
        and atr_rising
        and PROX_ATR_MIN <= prox <= PROX_ATR_MAX
        and vz >= VOL_Z_MIN_PRE
    )

    entry = breakout_level + BREAK_BUFFER_ATR * last_atr
    stop = last_low  # breakout bar low proxy on 1h
    t1 = entry + 0.8 * last_atr
    t2 = entry + 1.5 * last_atr

    out: Dict[str, Any] = {
        "symbol": symbol,
        "last": last_close,
        "atr": last_atr,
        "hh20": breakout_level,
        "ll20": ll20[-1],
        "entry": entry,
        "stop": stop,
        "t1": t1,
        "t2": t2,
        "vol_z": vz,
        "prox_atr": prox,
        "trend_ok": trend_ok,
        "lower_tf_ok": lower_tf_ok,
        "rs10": rs_strength,
        "score": score,
        "reasons": reasons,
    }

    if confirmed_breakout:
        out["signal"] = "B"
    elif pre_breakout:
        out["signal"] = "C"
    else:
        out["signal"] = "N"

    return out

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- MAIN --------------------------------------

def build_universe() -> Tuple[Dict[str, str], List[str]]:
    exch = fetch_exchange_usdt_symbols()
    mapping: Dict[str, str] = {}
    missing_binance: List[str] = []
    if not exch:
        return mapping, list(REVOLUT_TICKERS.keys())

    binance_symbols_available = set(exch.keys())
    tickers = [t for t in REVOLUT_TICKERS.keys() if t not in STABLES]
    for t in tickers:
        sym = f"{t}USDT"
        if sym in binance_symbols_available:
            mapping[t] = sym
        else:
            missing_binance.append(t)
    return mapping, missing_binance

def compute_regime(btc_1h: Dict[str, List[float]], btc_4h: Dict[str, List[float]]) -> Tuple[bool, str]:
    try:
        ema200_4h = ema(btc_4h["close"], EMA_SLOW)
        ema20_4h = ema(btc_4h["close"], EMA_FAST)
        ema200_1h = ema(btc_1h["close"], EMA_SLOW)
        ema20_1h = ema(btc_1h["close"], EMA_FAST)
        if len(ema200_4h) > 4 and len(ema200_1h) > 4:
            slope4 = ema20_4h[-1] - ema20_4h[-4]
            slope1 = ema20_1h[-1] - ema20_1h[-4]
            if (btc_4h["close"][-1] > ema200_4h[-1]) and (btc_1h["close"][-1] > ema200_1h[-1]) and slope4 > 0 and slope1 > 0:
                return True, "BTC uptrend (4h & 1h)"
            return False, "BTC not in uptrend"
        return False, "insufficient data"
    except Exception:
        return False, "regime calc error"

def run_pipeline(mode: str) -> dict:
    started = datetime.now(timezone.utc).astimezone(TZ)

    mapping, missing_binance = build_universe()
    if not mapping:
        payload = {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "exchangeInfo unavailable or no eligible symbols"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": sorted(missing_binance)}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }
        return payload

    # RS baseline: BTC
    btc_symbol = "BTCUSDT"
    btc_raw_1h = fetch_klines(btc_symbol, INTERVALS["1h"][0], INTERVALS["1h"][1])
    btc_raw_4h = fetch_klines(btc_symbol, INTERVALS["4h"][0], INTERVALS["4h"][1])

    if not btc_raw_1h or not btc_raw_4h:
        payload = {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "BTC data unavailable"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": len(mapping), "skipped": {"no_data": [], "missing_binance": sorted(missing_binance)}},
            "meta": {"binance_endpoint": BASE_URL, "mode": mode},
        }
        return payload

    btc_1h = parse_klines(btc_raw_1h)
    btc_4h = parse_klines(btc_raw_4h)
    regime_ok, regime_reason = compute_regime(btc_1h, btc_4h)
    btc_close_1h = btc_1h["close"]

    candidates: List[dict] = []
    confirmed: List[dict] = []
    no_data: List[str] = []
    scanned = 0

    # Optionally thin the mapping in light-fast mode for speed (scan subset)
    items = list(mapping.items())
    if mode == "light-fast":
        # simple downsample: every 2nd symbol
        items = items[::2]

    for t, sym in items:
        scanned += 1
        info = analyze_symbol(sym, btc_close_1h)
        time.sleep(0.03)  # light pacing
        if not info:
            no_data.append(t)
            continue
        info["ticker"] = t
        info["rotation_exempt"] = (t in ROTATION_EXEMPT)
        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_candidates = candidates[:MAX_CANDIDATES]

    signal_type = "HOLD"
    orders: List[dict] = []
    if confirmed:
        signal_type = "B"
        for o in confirmed:
            orders.append({
                "ticker": o["ticker"],
                "symbol": o["symbol"],
                "entry": round(o["entry"], 8),
                "stop": round(o["stop"], 8),
                "t1": round(o["t1"], 8),
                "t2": round(o["t2"], 8),
                "atr": round(o["atr"], 8),
                "tf": "1h",
                "notes": o.get("reasons", []),
                "rotation_exempt": o["rotation_exempt"],
            })
    elif top_candidates:
        signal_type = "C"

    cand_payload = [{
        "ticker": c["ticker"],
        "symbol": c["symbol"],
        "last": round(c["last"], 8),
        "atr": round(c["atr"], 8),
        "entry": round(c["entry"], 8),
        "stop": round(c["stop"], 8),
        "t1": round(c["t1"], 8),
        "t2": round(c["t2"], 8),
        "score": round(c["score"], 4),
        "vol_z": round(c["vol_z"], 2),
        "prox_atr": round(c["prox_atr"], 3),
        "rs10": round(c["rs10"], 4),
        "tf": "1h",
        "notes": c.get("reasons", []),
        "rotation_exempt": c["rotation_exempt"],
    } for c in top_candidates]

    payload: Dict[str, Any] = {
        "generated_at": human_time(started),
        "timezone": TZ_NAME,
        "regime": {"ok": regime_ok, "reason": regime_reason},
        "signals": {"type": signal_type},
        "orders": orders,
        "candidates": cand_payload,
        "universe": {
            "scanned": scanned,
            "eligible": len(mapping),
            "skipped": {
                "no_data": sorted(no_data),
                "missing_binance": sorted(missing_binance),
            },
        },
        "meta": {
            "params": {
                "ATR_LEN": ATR_LEN, "EMA_FAST": EMA_FAST, "EMA_SLOW": EMA_SLOW,
                "SWING_LOOKBACK": SWING_LOOKBACK,
                "PROX_ATR_MIN": PROX_ATR_MIN, "PROX_ATR_MAX": PROX_ATR_MAX,
                "VOL_Z_MIN_PRE": VOL_Z_MIN_PRE, "VOL_Z_MIN_BREAK": VOL_Z_MIN_BREAK,
                "BREAK_BUFFER_ATR": BREAK_BUFFER_ATR,
                "MAX_CANDIDATES": MAX_CANDIDATES
            },
            "lower_tf_confirm": True,
            "rs_reference": "BTCUSDT",
            "binance_endpoint": BASE_URL,
            "mode": mode,
        }
    }

    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses pipeline")
    parser.add_argument("--mode", default="deep", choices=["deep", "light-fast", "light-hourly"])
    parser.add_argument("--url", default="", help="reserved (unused)")
    args = parser.parse_args()

    try:
        payload = run_pipeline(args.mode)
    except Exception:
        log("UNCAUGHT ERROR:\n" + traceback.format_exc())
        payload = {
            "generated_at": human_time(),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "uncaught error"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            "meta": {"binance_endpoint": BASE_URL, "mode": args.mode},
        }

    latest_path = Path("public_runs/latest/summary.json")
    write_summary(payload, latest_path)

    stamp = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path("public_runs") / stamp
    ensure_dirs(snapshot_dir / "summary.json")
    with (snapshot_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

    log(f"Summary written to {latest_path} (signal={payload.get('signals',{}).get('type')}, regime_ok={payload.get('regime',{}).get('ok')})")

if __name__ == "__main__":
    main()