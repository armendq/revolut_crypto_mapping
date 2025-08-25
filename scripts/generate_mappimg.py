#!/usr/bin/env python3
"""
Generate a Revolut↔Binance symbol mapping.

Inputs (first one found wins):
- data/revolut_instruments.json   # [{"ticker":"BTC"}, {"ticker":"ETH"}, ...]
- data/revolut_tickers.csv        # CSV with column 'ticker'
- data/revolut_tickers.txt        # one ticker per line

Outputs:
- mapping/revolut_binance_mapping.json
- mapping/revolut_binance_unmatched.json     # Revolut tickers we couldn't match
- mapping/revolut_binance_report.txt         # human summary

Matching logic:
1) exact ticker match → {ticker}USDT exists on Binance
2) alias table (common exceptions) → e.g., ARBITRUM=ARB
3) fuzzy (rapidfuzz) on ticker vs Binance baseAsset (safe threshold)
"""

import json, csv, sys, os, re, time
from pathlib import Path
from typing import Dict, List, Tuple, Set
import requests

# ------------ config ------------
BINANCE_EXCHANGEINFO = "https://api.binance.com/api/v3/exchangeInfo"
OUT_DIR = Path("mapping")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR/"revolut_binance_mapping.json"
OUT_UNMATCHED = OUT_DIR/"revolut_binance_unmatched.json"
OUT_REPORT = OUT_DIR/"revolut_binance_report.txt"

# Try these Revolut inputs in order
REV_INPUTS = [
    Path("data/revolut_instruments.json"),
    Path("data/revolut_tickers.csv"),
    Path("data/revolut_tickers.txt"),
]

# Hand aliases (extend freely)
ALIASES: Dict[str, str] = {
    # Revolut -> Binance baseAsset
    "ARB": "ARB",           # Arbitrum
    "IMX": "IMX",
    "RNDR": "RNDR",
    "FET": "FET",
    "SUI": "SUI",
    "PEPE": "PEPE",
    "OP": "OP",
    "APT": "APT",
    "PYTH": "PYTH",
    "TIA": "TIA",
    "W": "W",
    "INJ": "INJ",
    "ATOM": "ATOM",
    "AVAX": "AVAX",
    "ADA": "ADA",
    "SOL": "SOL",
    "MATIC": "MATIC",
    "POL": "POL",           # Polygon 2.0 (if present)
    "DOGE": "DOGE",
    "SHIB": "SHIB",
    "DOT": "DOT",
    "ETH": "ETH",
    "BTC": "BTC",
    "BNB": "BNB",
    "LINK": "LINK",
    "NEAR": "NEAR",
    "APTOS": "APT",         # sometimes Revolut lists full name
    "ARBITRUM": "ARB",
    "POLKADOT": "DOT",
    "BITCOIN": "BTC",
    "ETHEREUM": "ETH",
    # add more long-name → ticker here if your Revolut source has names not tickers
}

# Symbols we should never map
BLOCKLIST: Set[str] = {
    "USDT", "BUSD", "FDUSD", "USDC", "DAI", "TUSD",
    "EUR", "GBP", "TRY", "BRL", "NGN", "RUB",
    "UST", "LUNA",  # legacy landmines
}

# Fuzzy matching (only if rapidfuzz is installed)
try:
    from rapidfuzz import fuzz, process
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False

FUZZ_THRESHOLD = 92  # require high confidence; adjust if needed

# ------------ helpers ------------
def load_revolut_tickers() -> List[str]:
    for p in REV_INPUTS:
        if p.exists():
            if p.suffix == ".json":
                with p.open() as f:
                    data = json.load(f)
                # Accept [{ticker:..}, or plain ["BTC",..]]
                if isinstance(data, list) and data and isinstance(data[0], dict) and "ticker" in data[0]:
                    tickers = [str(x["ticker"]).strip().upper() for x in data]
                else:
                    tickers = [str(x).strip().upper() for x in data]
                return [t for t in tickers if t and t not in BLOCKLIST]
            elif p.suffix == ".csv":
                out = []
                with p.open(newline="") as f:
                    r = csv.DictReader(f)
                    col = None
                    # find a suitable column
                    for c in r.fieldnames or []:
                        if c.lower() in ("ticker", "symbol", "revolut_ticker"):
                            col = c
                            break
                    if not col:
                        raise RuntimeError("CSV must have a 'ticker'/'symbol' column")
                    for row in r:
                        t = str(row[col]).strip().upper()
                        if t and t not in BLOCKLIST:
                            out.append(t)
                return out
            else:  # .txt
                out = []
                with p.open() as f:
                    for line in f:
                        t = line.strip().upper()
                        if t and t not in BLOCKLIST:
                            out.append(t)
                return out
    raise FileNotFoundError(
        "[generate_mapping] No Revolut tickers found. Provide one of: "
        "data/revolut_instruments.json, data/revolut_tickers.csv, data/revolut_tickers.txt"
    )

def fetch_binance_universe() -> Tuple[Set[str], Dict[str,str]]:
    """Return (baseAssets set, base->USDT symbol dict for spot)."""
    r = requests.get(BINANCE_EXCHANGEINFO, timeout=30)
    r.raise_for_status()
    info = r.json()
    bases: Set[str] = set()
    base_to_usdt: Dict[str, str] = {}
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        base = s.get("baseAsset", "").upper()
        quote = s.get("quoteAsset", "").upper()
        # Prefer USDT spot pairs as primary
        if quote == "USDT":
            base_to_usdt[base] = f"{base}{quote}"
        if base and base not in BLOCKLIST:
            bases.add(base)
    return bases, base_to_usdt

def try_exact(t: str, bases: Set[str], base_to_usdt: Dict[str,str]) -> Tuple[bool, str]:
    if t in bases and t in base_to_usdt:
        return True, base_to_usdt[t]
    return False, ""

def try_alias(t: str, bases: Set[str], base_to_usdt: Dict[str,str]) -> Tuple[bool, str]:
    a = ALIASES.get(t) or ALIASES.get(t.upper())
    if a and a in bases and a in base_to_usdt:
        return True, base_to_usdt[a]
    # If Revolut provided full name instead of ticker
    upper = re.sub(r"[^A-Z0-9]", "", t.upper())
    if upper in ALIASES:
        a2 = ALIASES[upper]
        if a2 in bases and a2 in base_to_usdt:
            return True, base_to_usdt[a2]
    return False, ""

def try_fuzzy(t: str, bases: Set[str], base_to_usdt: Dict[str,str]) -> Tuple[bool, str, int, str]:
    if not HAVE_FUZZ:
        return False, "", 0, ""
    choices = list(bases)
    match, score, idx = process.extractOne(t, choices, scorer=fuzz.WRatio)
    if score >= FUZZ_THRESHOLD and match in base_to_usdt:
        return True, base_to_usdt[match], score, match
    return False, "", score, match if 'match' in locals() else ""

def main() -> int:
    revolut = load_revolut_tickers()
    bases, base2usdt = fetch_binance_universe()

    mapping = []
    unmatched = []
    report_lines = []
    seen_revolut = set()

    for t in revolut:
        if t in seen_revolut:
            continue
        seen_revolut.add(t)

        ok, sym = try_exact(t, bases, base2usdt)
        if ok:
            mapping.append({"revolut_ticker": t, "binance_symbol": sym})
            continue

        ok, sym = try_alias(t, bases, base2usdt)
        if ok:
            mapping.append({"revolut_ticker": t, "binance_symbol": sym})
            continue

        ok, sym, sc, guess = try_fuzzy(t, bases, base2usdt)
        if ok:
            mapping.append({"revolut_ticker": t, "binance_symbol": sym})
            report_lines.append(f"[FUZZY] {t} → {guess} ({sym}) score={sc}")
        else:
            unmatched.append(t)

    # Sort deterministically
    mapping.sort(key=lambda x: x["revolut_ticker"])
    with OUT_JSON.open("w") as f:
        json.dump(mapping, f, indent=2)

    with OUT_UNMATCHED.open("w") as f:
        json.dump(unmatched, f, indent=2)

    report_lines.insert(0, f"Revolut tickers: {len(revolut)}")
    report_lines.insert(1, f"Matched: {len(mapping)}")
    report_lines.insert(2, f"Unmatched: {len(unmatched)}")
    if unmatched:
        report_lines.append("\nUnmatched:")
        report_lines += sorted(unmatched)

    with OUT_REPORT.open("w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[generate_mapping] matched {len(mapping)} / {len(revolut)}; "
          f"unmatched {len(unmatched)} → see {OUT_UNMATCHED}")
    return 0

if __name__ == "__main__":
    sys.exit(main())