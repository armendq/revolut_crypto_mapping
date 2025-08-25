#!/usr/bin/env python3
"""
Generate a Revolut ↔ Binance USDT-spot mapping from your provided dict.

INPUT (required):
  data/revolut_mapping.json    # exactly the dict you pasted: {"BTC":"Bitcoin", ...}

OUTPUTS:
  mapping/revolut_binance_mapping.json         # [{revolut_ticker, binance_symbol}]
  mapping/revolut_binance_unmatched.json       # ["TICKER", ...]
  mapping/revolut_binance_report.txt           # human-readable summary

Matching rules (in order):
  1) Exact ticker = Binance baseAsset and USDT spot pair exists → BASEUSDT
  2) Alias table (common differences, long names, renamed tickers)
  3) Optional fuzzy match (rapidfuzz) on ticker or name vs Binance baseAsset (strict threshold)

Notes:
  - We only map to USDT spot pairs (clean & most liquid).
  - Stablecoins / fiat symbols are ignored (USDT/USDC/DAI/...).
"""

from __future__ import annotations
import json, sys, re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import requests

# ---------- config ----------
REV_FILE = Path("data/revolut_mapping.json")
OUT_DIR = Path("mapping"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "revolut_binance_mapping.json"
OUT_UNMATCHED = OUT_DIR / "revolut_binance_unmatched.json"
OUT_REPORT = OUT_DIR / "revolut_binance_report.txt"

BINANCE_EXCHANGEINFO = "url = "url = "https://data-api.binance.vision/api/v3/exchangeInfo"

BLOCKLIST: Set[str] = {
    "USDT","USDC","FDUSD","BUSD","TUSD","DAI",
    "EUR","GBP","TRY","BRL","NGN","RUB"
}

# Common divergences (Revolut ticker or name → Binance baseAsset)
ALIASES: Dict[str, str] = {
    # Obvious/legacy/name differences
    "POLYGON": "MATIC",
    "CELESTIA": "TIA",
    "AVALANCHE": "AVAX",
    "COSMOS": "ATOM",
    "AAVE": "AAVE",
    "AIOZ NETWORK": "AIOZ",
    "ARWEAVE": "AR",
    "AKASH NETWORK": "AKT",
    "BICONOMY": "BICO",
    "BIFROST": "BNC",
    "BRAINTRUST": "BTRST",
    "CARTESI": "CTSI",
    "CELER NETWORK": "CELR",
    "CIVIC": "CVC",
    "CLOVER FINANCE": "CLV",
    "CONFLUX": "CFX",
    "COVALENT": "CQT",
    "CRONOS": "CRO",
    "CURVE DAO TOKEN": "CRV",
    "DENT": "DENT",
    "DIGIBYTE": "DGB",
    "DOGECOIN": "DOGE",
    "ETHEREUM": "ETH",
    "ETHEREUM CLASSIC": "ETC",
    "ETHEREUMPOW": "ETHW",
    "FANTOM": "FTM",
    "FILECOIN": "FIL",
    "FLOKI": "FLOKI",
    "FRAx SHARE": "FXS",
    "GAINS NETWORK": "GNS",
    "GODS UNCHAINED": "GODS",
    "GNOSIS": "GNO",
    "HEDERA": "HBAR",
    "ILLUVIUM": "ILV",
    "IMMUTABLE": "IMX",
    "INJECTIVE": "INJ",
    "IEXEC RLC": "RLC",
    "JASMYCOIN": "JASMY",
    "KADENA": "KDA",
    "KYBER NETWORK": "KNC",
    "KUSAMA": "KSM",
    "LIDO DAO": "LDO",
    "LIVEPEER": "LPT",
    "LOOPRING": "LRC",
    "MASK NETWORK": "MASK",
    "MERIT CIRCLE": "MC",
    "MINA": "MINA",
    "MOONBEAM": "GLMR",
    "NEAR PROTOCOL": "NEAR",
    "OASIS NETWORK": "ROSE",
    "OCEAN PROTOCOL": "OCEAN",
    "OND0": "ONDO",  # common OCR mistake
    "ONTOLOGY": "ONT",
    "ONTOLOGY GAS": "ONG",
    "OPTIMISM": "OP",
    "PAX GOLD": "PAXG",
    "PENDLE": "PENDLE",
    "PIXELS": "PIXEL",
    "POLYMESH": "POLYX",
    "POCKET NETWORK": "POKT",
    "POWERLEDGER": "POWR",
    "PUNDI X": "PUNDIX",
    "RAYDIUM": "RAY",
    "RENDER": "RNDR",
    "RESERVE RIGHTS": "RSR",
    "ROCKET POOL": "RPL",
    "SHIBA INU": "SHIB",
    "SKALE NETWORK": "SKL",
    "SMOOTH LOVE POTION": "SLP",
    "SOLANA": "SOL",
    "STARGATE FINANCE": "STG",
    "STARKNET": "STRK",
    "STORJ": "STORJ",
    "SUI": "SUI",
    "SUSHISWAP": "SUSHI",
    "SYNAPSE": "SYN",
    "TELLOR": "TRB",
    "THE GRAPH": "GRT",
    "THE SANDBOX": "SAND",
    "THRESHOLD": "T",
    "TONCOIN": "TON",
    "TRON": "TRX",
    "UNISWAP": "UNI",
    "VECHAIN": "VET",
    "VETHOR TOKEN": "VTHO",
    "VOXIES": "VOXEL",
    "WORLDCOIN": "WLD",
    "WOO NETWORK": "WOO",
    "WRAPPED BITCOIN": "WBTC",
    "WRAPPED CENTRIFUGE": "WCFG",
    "YIELD GUILD GAMES": "YGG",
    "0X": "ZRX",

    # Ticker-level special cases (Revolut → Binance)
    "WBTC": "WBTC",
    "WAXP": "WAXP",
    "WCFG": "WCFG",
    "SYNTH": "SYN",
}

# ---------- optional fuzzy ----------
try:
    from rapidfuzz import process, fuzz
    HAVE_FUZZ = True
    FUZZ_THRESHOLD = 92
except Exception:
    HAVE_FUZZ = False

def load_revolut() -> Dict[str, str]:
    if not REV_FILE.exists():
        raise FileNotFoundError(f"Missing {REV_FILE} with Revolut dict.")
    data = json.loads(REV_FILE.read_text())
    # normalize to UPPER keys, UPPER names for matching
    out = {}
    for k, v in data.items():
        k2 = str(k).strip().upper()
        if not k2 or k2 in BLOCKLIST:
            continue
        out[k2] = str(v).strip()
    return out

def fetch_binance() -> Tuple[Set[str], Dict[str, str]]:
    """Return (baseAssets set, base->USDT symbol mapping)."""
    r = requests.get(BINANCE_EXCHANGEINFO, timeout=30)
    r.raise_for_status()
    info = r.json()
    bases: Set[str] = set()
    base2usdt: Dict[str, str] = {}
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        base = s.get("baseAsset", "").upper()
        quote = s.get("quoteAsset", "").upper()
        if not base or base in BLOCKLIST:
            continue
        bases.add(base)
        if quote == "USDT":
            base2usdt[base] = f"{base}{quote}"
    return bases, base2usdt

def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", s.upper())

def try_exact(t: str, bases: Set[str], base2usdt: Dict[str,str]) -> str | None:
    if t in bases and t in base2usdt:
        return base2usdt[t]
    return None

def try_alias(t: str, name: str, bases: Set[str], base2usdt: Dict[str,str]) -> str | None:
    # ticker alias
    a = ALIASES.get(t) or ALIASES.get(clean(t))
    if a and a in base2usdt:
        return base2usdt[a]
    # name alias
    nm = ALIASES.get(name.upper()) or ALIASES.get(clean(name))
    if nm and nm in base2usdt:
        return base2usdt[nm]
    return None

def try_fuzzy(t: str, name: str, bases: Set[str], base2usdt: Dict[str,str]) -> Tuple[str | None, int, str]:
    if not HAVE_FUZZ:
        return None, 0, ""
    choices = list(bases)
    # try ticker first, then name
    cand1 = process.extractOne(t, choices, scorer=fuzz.WRatio)
    cand2 = process.extractOne(clean(name), choices, scorer=fuzz.WRatio) if name else None
    best = None
    if cand1 and (not cand2 or cand1[1] >= cand2[1]): best = cand1
    elif cand2: best = cand2
    if best and best[1] >= FUZZ_THRESHOLD and best[0] in base2usdt:
        return base2usdt[best[0]], best[1], best[0]
    return None, 0, ""

def main() -> int:
    rev = load_revolut()                   # {"BTC":"Bitcoin", ...}
    bases, base2usdt = fetch_binance()     # Binance spot

    mapping = []
    unmatched = []
    report = []

    for tkr, nm in sorted(rev.items()):
        # 1) exact
        sym = try_exact(tkr, bases, base2usdt)
        if sym:
            mapping.append({"revolut_ticker": tkr, "binance_symbol": sym})
            continue
        # 2) alias
        sym = try_alias(tkr, nm, bases, base2usdt)
        if sym:
            mapping.append({"revolut_ticker": tkr, "binance_symbol": sym})
            continue
        # 3) fuzzy (strict)
        sym, score, guess = try_fuzzy(tkr, nm, bases, base2usdt)
        if sym:
            mapping.append({"revolut_ticker": tkr, "binance_symbol": sym})
            report.append(f"[FUZZY] {tkr} / {nm} → {guess} ({sym}) score={score}")
            continue
        # no luck
        unmatched.append(tkr)

    # write outputs
    mapping.sort(key=lambda x: x["revolut_ticker"])
    OUT_JSON.write_text(json.dumps(mapping, indent=2))
    OUT_UNMATCHED.write_text(json.dumps(unmatched, indent=2))

    report.insert(0, f"Revolut tickers: {len(rev)}")
    report.insert(1, f"Matched: {len(mapping)}")
    report.insert(2, f"Unmatched: {len(unmatched)}")
    if unmatched:
        report.append("\nUnmatched:")
        report += unmatched
    OUT_REPORT.write_text("\n".join(report) + "\n")

    print(f"[generate_mapping] matched {len(mapping)} / {len(rev)}; "
          f"unmatched {len(unmatched)} (see {OUT_UNMATCHED})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
