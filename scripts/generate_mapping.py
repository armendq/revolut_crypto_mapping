#!/usr/bin/env python3
# scripts/generate_mapping.py
"""
Generate Revolut ↔ Binance symbol mapping.

Inputs
------
- data/revolut_list.json : {"BTC":"Bitcoin", ...}

Outputs
-------
- data/revolut_binance_mapping.json : [
    {
      "revolut_ticker": "BTC",
      "revolut_name": "Bitcoin",
      "binance_symbol": "BTCUSDT",
      "binance_base": "BTC",
      "binance_quote": "USDT"
    },
    ...
  ]
- data/revolut_binance_mapping.csv    (same data, CSV)
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# --------------------------- Config ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = REPO_ROOT / "data"

REV_FILE   = DATA_DIR / "revolut_list.json"
OUT_JSON   = DATA_DIR / "revolut_binance_mapping.json"
OUT_CSV    = DATA_DIR / "revolut_binance_mapping.csv"

# Binance Vision mirror avoids 451 from api.binance.com
BINANCE_BASE = "https://data-api.binance.vision/api/v3"
BINANCE_EXCHANGEINFO = f"{BINANCE_BASE}/exchangeInfo"

# We prefer quotes in this priority
QUOTE_PRIORITY = ("USDT", "FDUSD", "USDC")

# Known ticker differences or special choices:
# key = Revolut ticker, value = preferred Binance base asset (or full symbol with quote)
EXPLICIT_OVERRIDES: Dict[str, str] = {
    # Revolut -> Binance
    # Wrapped Centrifuge: Revolut WCFG, Binance base is CFG
    "WCFG": "CFG",
    # Merit Circle → MC (Binance base is still MC)
    # POKT is not on Binance spot (will be skipped), included here for awareness
    # ETHW delisted on Binance spot (skip)
    # Add more here as you hit mismatches:
    # "MATIC": "MATIC",  # example (identity, not needed)
}

# HTTP/retry
TIMEOUT = 20
MAX_TRIES = 4
BACKOFF_SECONDS = (1.0, 2.0, 3.5, 5.0)

# --------------------------- Logging ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[generate-mapping] %(message)s",
)
log = logging.getLogger("generate-mapping")


# --------------------------- Data types ---------------------------

@dataclass
class MapRow:
    revolut_ticker: str
    revolut_name: str
    binance_symbol: str
    binance_base: str