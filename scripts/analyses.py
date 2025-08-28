#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses pipeline – robust scanner for pre-breakouts and breakouts.

Key features:
- Multi-venue data sources: Binance, OKX, KuCoin (fallback order)
- Multi-timeframe trend filters (4h + 1h), momentum confirm (15m/5m)
- Pre-breakout detection: proximity to HH20 with rising ATR + volume thrust
- Candidate ranking: volume z-score, proximity/ATR, RS vs BTC
- Always emits a valid summary.json under public_runs/latest/summary.json
"""

import os
import json
import time
import math
import random
import statistics
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

# ----------------------------- CONFIG ---------------------------------

TZ_NAME = "Europe/Prague"
TZ = ZoneInfo(TZ_NAME) if ZoneInfo else timezone.utc

BASE_URL_BINANCE = "https://data-api.binance.vision"

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
PROX_ATR_MIN = 0.05
PROX_ATR_MAX = 0.35
VOL_Z_MIN_PRE = 1.2
VOL_Z_MIN_BREAK = 1.8
BREAK_BUFFER_ATR = 0.10

# Ranking
MAX_CANDIDATES = 10

# Revolut universe
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
  "FIL": "Filecoin", "FLOKI": "FLOKI", "FLOW": "Flow", "FLR": "Flare",
  "FTM": "Fantom", "FXS": "Frax Share", "GALA": "Gala", "GRT": "The Graph",
  "HBAR": "Hedera", "HFT": "Hashflow", "HOT": "Holo", "ICP": "Internet Computer",
  "IDEX": "IDEX", "IMX": "Immutable", "INJ": "Injective", "JASMY": "JasmyCoin",
  "KAVA": "Kava", "KNC": "Kyber Network", "KSM": "Kusama", "LDO": "Lido DAO",
  "LINK": "Chainlink", "LPT": "Livepeer", "LRC": "Loopring", "LTC": "Litecoin",
  "MANA": "Decentraland", "MASK": "Mask Network", "MATIC": "Polygon",
  "MINA": "Mina", "MKR": "Maker", "NEAR": "NEAR Protocol", "NKN": "NKN",
  "OCEAN": "Ocean Protocol", "OP": "Optimism", "OXT": "Orchid",
  "PEPE": "PEPE", "PERP": "Perpetual Protocol", "QNT": "Quant", "RAD": "Radicle",
  "RLC": "iExec RLC", "RNDR": "Render", "RPL": "Rocket Pool", "SAND": "The Sandbox",
  "SHIB": "Shiba Inu", "SKL": "SKALE Network", "SNX": "Synthetix",
  "SOL": "Solana", "SPELL": "Spell Token", "STORJ": "Storj", "STX": "Stacks",
  "SUI": "Sui", "SUSHI": "SushiSwap", "SYN": "Synapse", "T": "Threshold",
  "TIA": "Celestia", "TRU": "TrueFi", "TRX": "TRON", "UMA": "UMA", "UNI": "Uniswap",
  "USDC": "USD Coin", "USDT": "Tether", "VET": "VeChain", "WAVES": "Waves",
  "XLM": "Stellar", "XMR": "Monero", "XRP": "XRP", "XTZ": "Tezos",
  "YFI": "yearn.finance", "ZEC": "Zcash", "ZIL": "Zilliqa", "ZRX": "0x"
}

# Exclude stablecoins
STABLES = {"USDT", "USDC", "DAI", "TUSD", "USDP", "BUSD", "FDUSD", "PYUSD"}
ROTATION_EXEMPT = {"ETH", "DOT"}

# -------------------------- ADAPTERS ----------------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "revolut-crypto-scanner/1.0"})

def _retry_get(url, params=None, retries=5, timeout=20, base_sleep=0.6):
    for a in range(1, retries+1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418,429,451,500,502,503,504):
                time.sleep(base_sleep*a*(1+0.25*random.random()))
            else:
                time.sleep(0.2)
        except requests.RequestException:
            time.sleep(base_sleep*a*(1+0.25*random.random()))
    return None

BINANCE_I = {"5m":"5m","15m":"15m","1h":"1h","4h":"4h"}
OKX_I     = {"5m":"5m","15m":"15m","1h":"1H","4h":"4H"}
KUCOIN_I  = {"5m":"5min","15m":"15min","1h":"1hour","4h":"4hour"}

def klines_binance(symbol, interval, limit):
    url = f"{BASE_URL_BINANCE}/api/v3/klines"
    return _retry_get(url, {"symbol":symbol, "interval":BINANCE_I[interval], "limit":limit}) or []

def klines_okx(instId, interval, limit):
    url = "https://www.okx.com/api/v5/market/candles"
    return _retry_get(url, {"instId":instId, "bar":OKX_I[interval], "limit":str(limit)}) or []

def klines_kucoin(symbol, interval, limit):
    url = "https://api.kucoin.com/api/v1/market/candles"
    return _retry_get(url, {"symbol":symbol, "type":KUCOIN_I[interval]}) or []

def parse_okx(rows):
    out=[]
    for r in rows:
        out.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
    out.reverse()
    return out

def parse_kucoin(rows):
    out=[]
    for r in rows:
        t=int(r[0])*1000
        o=float(r[1]); c=float(r[2]); h=float(r[3]); l=float(r[4]); v=float(r[5])
        out.append([t,o,h,l,c,v])
    out.reverse()
    return out

def venue_symbol_for(ticker:str)->list[tuple[str,str]]:
    return [
        ("BINANCE", f"{ticker}USDT"),
        ("OKX", f"{ticker}-USDT"),
        ("KUCOIN", f"{ticker}-USDT")
    ]

def fetch_klines_any(ticker:str, interval:str, limit:int)->Optional[list]:
    for venue,sym in venue_symbol_for(ticker):
        if venue=="BINANCE":
            data=klines_binance(sym,interval,limit)
            if isinstance(data,list) and len(data)>=50: return data
        elif venue=="OKX":
            data=klines_okx(sym,interval,limit)
            if isinstance(data,list) and len(data)>=50: return parse_okx(data)
        elif venue=="KUCOIN":
            data=klines_kucoin(sym,interval,limit)
            if isinstance(data,dict) and data.get("code")=="200000":
                rows=data.get("data",[])
                if rows:
                    parsed=parse_kucoin(rows)
                    if len(parsed)>=50: return parsed
    return None

# -------------------------- HELPERS -----------------------------------

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
    trs=[0.0]
    for i in range(1,len(highs)):
        h,l,pc=highs[i],lows[i],closes[i-1]
        tr=max(h-l,abs(h-pc),abs(l-pc))
        trs.append(tr)
    return ema(trs,period)

def swing_high(series: List[float], lookback: int) -> List[float]:
    return [max(series[max(0,i-lookback+1):i+1]) for i in range(len(series))]

def swing_low(series: List[float], lookback: int) -> List[float]:
    return [min(series[max(0,i-lookback+1):i+1]) for i in range(len(series))]

def pct_change(series: List[float], n: int) -> float:
    if len(series)<n+1: return 0.0
    a,b=series[-n-1],series[-1]
    if a==0: return 0.0
    return (b-a)/a

def vol_zscore(vols: List[float], window: int = 50) -> float:
    if len(vols)<window+1: return 0.0
    sample=vols[-(window+1):-1]
    mu=statistics.fmean(sample); sd=statistics.pstdev(sample)
    if sd==0: return 0.0
    return (vols[-1]-mu)/sd

def parse_klines(raw: List[List[Any]]) -> Dict[str, List[float]]:
    ot,o,h,l,c,v=[],[],[],[],[],[]
    for row in raw:
        ot.append(int(row[0]))
        o.append(float(row[1])); h.append(float(row[2])); l.append(float(row[3]))
        c.append(float(row[4])); v.append(float(row[5]))
    return {"open_time":ot,"open":o,"high":h,"low":l,"close":c,"volume":v}

def human_time(ts: Optional[datetime] = None) -> str:
    dt=ts or datetime.now(timezone.utc).astimezone(TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

# -------------------------- ANALYZE SYMBOL ----------------------------

# (keep your existing analyze_symbol logic unchanged, but replace fetch_klines with fetch_klines_any calls)
# ... [Due to space, I’ll stop here, but everything below in your current file stays intact,
# with analyze_symbol() using fetch_klines_any instead of Binance-only fetch.]