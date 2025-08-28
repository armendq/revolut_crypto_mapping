#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline – robust scanner for pre-breakouts and breakouts.

Key features:
- Uses data-api.binance.vision (works in restricted regions)
- Never calls Binance without a symbol (avoids 451s)
- Multi-timeframe trend filters (4h + 1h), momentum confirm (15m/5m)
- Pre-breakout detection: proximity to HH20 with rising ATR + volume thrust
- Candidate ranking: volume z-score, proximity/ATR, RS vs BTC
- Always emits a valid summary.json under public_runs/latest/summary.json
- Adds robust HTTP retries, explicit logging, and hard failures on empty universe
"""

import os
import sys
import json
import time
import math
import argparse
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
PROX_ATR_MIN = 0.05   # close must be at least this many ATR below HH20 to be considered "pre" (avoid already-broken)
PROX_ATR_MAX = 0.35   # within 0.35 ATR of HH20
VOL_Z_MIN_PRE = 1.2   # volume z-score for pre-breakout
VOL_Z_MIN_BREAK = 1.8 # stronger thrust needed for confirmed breakout
BREAK_BUFFER_ATR = 0.10  # entry buffer above HH20 for B signals

# Ranking
MAX_CANDIDATES = 10

# Revolut universe (keep full, we’ll filter by Binance availability)
REVOLUT_TICKERS = {
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

# ----------------------------- LOGGING & HTTP SESSION ------------------

def log(msg: str) -> None:
    print(f"[analyses] {msg}", flush=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "revolut-crypto-analyses/1.0"})
_retry = Retry(
    total=6,
    connect=3,
    read=3,
    backoff_factor=0.6,
    status_forcelist=[418, 429, 451, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.mount("http://", HTTPAdapter(max_retries=_retry))

def safe_get(url: str, params: Optional[dict] = None, retries: int = 0, timeout: int = 20, sleep_s: float = 0.6) -> Optional[requests.Response]:
    """GET with robust retries/backoff via shared SESSION. Returns Response or None."""
    for attempt in range(1, 4):  # outer loop on top of Retry
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            log(f"GET {url} -> {r.status_code}; attempt {attempt}")
            time.sleep(sleep_s * attempt)
        except requests.RequestException as e:
            log(f"GET {url} exception: {e}; attempt {attempt}")
            time.sleep(sleep_s * attempt)
    return None

# -------------------------- INDICATORS --------------------------------

def ema(series: List[float], period: int) -> List[float]:
    if not series or period <= 1:
        return series[:]
    alpha = 2.0 / (period + 1.0)
    out = []
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
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return ema(trs, period)

def swing_high(series: List[float], lookback: int) -> List[float]:
    out = []
    for i in range(len(series)):
        if i < lookback:
            out.append(max(series[:i+1]))
        else:
            out.append(max(series[i - lookback + 1:i+1]))
    return out

def swing_low(series: List[float], lookback: int) -> List[float]:
    out = []
    for i in range(len(series)):
        if i < lookback:
            out.append(min(series[:i+1]))
        else:
            out.append(min(series[i - lookback + 1:i+1]))
    return out

def pct_change(series: List[float], n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    a, b = series[-n-1], series[-1]
    if a == 0:
        return 0.0
    return (b - a) / a

def vol_zscore(vols: List[float], window: int = 50) -> float:
    if len(vols) < window + 1:
        return 0.0
    sample = vols[-(window+1):-1]
    mu = statistics.fmean(sample)
    sd = statistics.pstdev(sample)
    if sd == 0:
        return 0.0
    return (vols[-1] - mu) / sd

# -------------------------- DATA FETCHERS ------------------------------

def fetch_exchange_usdt_symbols() -> Dict[str, dict]:
    """Return dict of USDT spot symbols metadata keyed by symbol (e.g., 'LPTUSDT')."""
    url = f"{BASE_URL}/api/v3/exchangeInfo"
    r = safe_get(url, retries=3, timeout=25)
    if not r:
        return {}
    try:
        data = r.json()
    except Exception:
        return {}
    out = {}
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
    ot, o, h, l, c, v = [], [], [], [], [], []
    for row in raw:
        # [ openTime, open, high, low, close, volume, closeTime, ... ]
        ot.append(int(row[0]))
        o.append(float(row[1]))
        h.append(float(row[2]))
        l.append(float(row[3]))
        c.append(float(row[4]))
        v.append(float(row[5]))
    return {"open_time": ot, "open": o, "high": h, "low": l, "close": c, "volume": v}

def human_time(ts: Optional[datetime] = None) -> str:
    dt = ts or datetime.now(timezone.utc).astimezone(TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

# -------------------------- CORE ANALYTICS -----------------------------

def analyze_symbol(symbol: str, btc_1h_close: List[float]) -> Optional[dict]:
    """Analyze single symbol; return candidate dict or confirmed breakout dict with 'signal' key."""
    # Fetch required timeframes
    req = {}
    for tf, (interval, limit) in INTERVALS.items():
        raw = fetch_klines(symbol, interval, limit)
        if not raw or len(raw) < max(EMA_SLOW + 5, SWING_LOOKBACK + 5):
            return None
        req[tf] = parse_klines(raw)

    # Extract series
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

    # EMA20 slope > 0 (last vs 3 bars ago)
    ema20_slope_1h = ema20_1h[-1] - ema20_1h[-4]
    ema20_slope_4h = ema20_4h[-1] - ema20_4h[-4]

    trend_ok = (
        ema20_slope_1h > 0 and ema20_slope_4h > 0 and
        close_1h[-1] > ema200_1h[-1] and four["close"][-1] > ema200_4h[-1]
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
    prox = (hh20[-1] - close_1h[-1]) / max(1e-9, atr_1h[-1])  # in ATRs above price
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
    last_high = high_1h[-1]
    last_atr = atr_1h[-1]

    # Build candidate info
    reasons = []
    if trend_ok:
        reasons.append("TrendOK(1h&4h)")
    if atr_rising:
        reasons.append("ATR↑")
    if vz > 0:
        reasons.append(f"VolZ={vz:.2f}")
    if lower_tf_ok:
        reasons.append("LowerTF OK")

    # Scores for ranking candidates (pre-breakouts)
    score = 0.0
    if vz > 0:
        score += vz
    if prox > 0:
        score += 1.0 / (min(max(prox, 0.01), 2.0))  # closer to HH improves score
    score += max(0.0, 5.0 * rs_strength)  # boost RS
    if lower_tf_ok:
        score += 0.5

    # Determine signals
    breakout_level = hh20[-1]
    confirmed_breakout = (
        trend_ok and
        last_close > (breakout_level + BREAK_BUFFER_ATR * last_atr) and
        vz >= VOL_Z_MIN_BREAK and
        lower_tf_ok
    )

    pre_breakout = (
        trend_ok and atr_rising and
        PROX_ATR_MIN <= prox <= PROX_ATR_MAX and
        vz >= VOL_Z_MIN_PRE
    )

    # Proposed trading parameters (entry/stop)
    entry = breakout_level + BREAK_BUFFER_ATR * last_atr
    stop = last_low  # breakout bar low as downstream rule; here we use last 1h low proxy
    t1 = entry + 0.8 * last_atr
    t2 = entry + 1.5 * last_atr

    out = {
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

# -------------------------- IO HELPERS --------------------------------

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_summary(payload: dict, dest_latest: Path) -> None:
    ensure_dirs(dest_latest)
    with dest_latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

# -------------------------- MAIN --------------------------------------

def main() -> None:
    started = datetime.now(timezone.utc).astimezone(TZ)
    t0 = time.time()
    log(f"start {human_time(started)}")

    # CLI (mode only for consistency with workflow; logic is unchanged)
    try:
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--mode", default="auto")
        ap.add_argument("--debug", action="store_true")
        args, _ = ap.parse_known_args()
    except Exception:
        class _A: mode="auto"; debug=False
        args=_A()

    # 1) exchangeInfo
    exch = fetch_exchange_usdt_symbols()
    if not exch:
        payload = {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "exchangeInfo unavailable"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
        }
        write_summary(payload, Path("public_runs/latest/summary.json"))
        log("FATAL: exchangeInfo unavailable -> wrote HOLD summary and exiting non-zero.")
        sys.exit(2)

    log(f"exchange symbols loaded: {len(exch)} in {time.time()-t0:.2f}s")

    # 2) build mapping (Revolut tickers -> Binance USDT symbols)
    binance_symbols_available = set(exch.keys())
    tickers = [t for t in REVOLUT_TICKERS.keys() if t not in STABLES]
    mapping: Dict[str, str] = {}
    missing_binance: List[str] = []
    for t in tickers:
        sym = f"{t}USDT"
        if sym in binance_symbols_available:
            mapping[t] = sym
        else:
            missing_binance.append(t)

    if len(mapping) == 0:
        payload = {
            "generated_at": human_time(started),
            "timezone": TZ_NAME,
            "regime": {"ok": False, "reason": "no eligible symbols after mapping"},
            "signals": {"type": "HOLD"},
            "orders": [],
            "candidates": [],
            "universe": {
                "scanned": 0,
                "eligible": 0,
                "skipped": {"no_data": [], "missing_binance": sorted(missing_binance)},
            },
        }
        write_summary(payload, Path("public_runs/latest/summary.json"))
        log(f"FATAL: eligible mapping is 0; missing examples: {missing_binance[:10]}")
        sys.exit(3)

    log(f"eligible mapping: {len(mapping)} (first 10: {list(mapping.items())[:10]})")

    # 3) prefetch BTC for RS reference
    t1 = time.time()
    btc_symbol = "BTCUSDT"
    btc_raw_1h = fetch_klines(btc_symbol, INTERVALS["1h"][0], INTERVALS["1h"][1])
    if not btc_raw_1h:
        log("WARN: BTC 1h klines unavailable; RS will be disabled.")
        btc_close_1h: List[float] = []
    else:
        btc_close_1h = parse_klines(btc_raw_1h)["close"]
    log(f"BTC prefetch done in {time.time()-t1:.2f}s")

    # Market regime from BTC (4h + 1h)
    regime_ok = False
    regime_reason = "insufficient data"
    try:
        btc_raw_4h = fetch_klines(btc_symbol, INTERVALS["4h"][0], INTERVALS["4h"][1])
        if btc_raw_4h and btc_raw_1h:
            btc_4h = parse_klines(btc_raw_4h)
            btc_1h = parse_klines(btc_raw_1h)
            ema200_4h = ema(btc_4h["close"], EMA_SLOW)
            ema20_4h = ema(btc_4h["close"], EMA_FAST)
            ema200_1h = ema(btc_1h["close"], EMA_SLOW)
            ema20_1h = ema(btc_1h["close"], EMA_FAST)
            if len(ema200_4h) > 4 and len(ema200_1h) > 4:
                slope4 = ema20_4h[-1] - ema20_4h[-4]
                slope1 = ema20_1h[-1] - ema20_1h[-4]
                if (btc_4h["close"][-1] > ema200_4h[-1]) and (btc_1h["close"][-1] > ema200_1h[-1]) and slope4 > 0 and slope1 > 0:
                    regime_ok = True
                    regime_reason = "BTC uptrend (4h & 1h)"
                else:
                    regime_reason = "BTC not in uptrend"
    except Exception as e:
        log(f"WARN: regime calc error: {e}")

    # 4) scan symbols
    t_scan = time.time()
    candidates: List[dict] = []
    confirmed: List[dict] = []
    no_data: List[str] = []
    scanned = 0

    for idx, (tkr, sym) in enumerate(mapping.items(), 1):
        scanned += 1
        if args.debug and idx % 25 == 0:
            log(f"scanning {idx}/{len(mapping)}…")
        info = analyze_symbol(sym, btc_close_1h)
        time.sleep(0.05)  # gentle pacing
        if not info:
            no_data.append(tkr)
            continue

        # attach annotations
        info["ticker"] = tkr
        info["rotation_exempt"] = (tkr in ROTATION_EXEMPT)

        if info["signal"] == "B":
            confirmed.append(info)
        elif info["signal"] == "C":
            candidates.append(info)

    log(f"scan done: scanned={scanned}, confirmed={len(confirmed)}, candidates={len(candidates)} in {time.time()-t_scan:.1f}s")

    # Rank candidates
    candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    top_candidates = candidates[:MAX_CANDIDATES]

    # Signals
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

    # Build candidates payload
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

    payload = {
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
        }
    }

    # Write outputs
    latest_path = Path("public_runs/latest/summary.json")
    write_summary(payload, latest_path)

    # Also write a timestamped snapshot for auditing/reruns
    stamp = datetime.now(timezone.utc).astimezone(TZ).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path("public_runs") / stamp
    ensure_dirs(snapshot_dir / "summary.json")
    with (snapshot_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)

    log(f"Summary written: {latest_path} (signal={signal_type}, regime_ok={regime_ok}); total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log("UNCAUGHT ERROR:\n" + traceback.format_exc())
        # Write a minimal HOLD so downstream fetchers don’t 404
        try:
            payload = {
                "generated_at": human_time(),
                "timezone": TZ_NAME,
                "regime": {"ok": False, "reason": "uncaught error"},
                "signals": {"type": "HOLD"},
                "orders": [],
                "candidates": [],
                "universe": {"scanned": 0, "eligible": 0, "skipped": {"no_data": [], "missing_binance": []}},
            }
            write_summary(payload, Path("public_runs/latest/summary.json"))
        finally:
            sys.exit(1)
```0