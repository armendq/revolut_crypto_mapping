#!/usr/bin/env python3
import csv, json, time, re
from pathlib import Path
from typing import Dict, List, Optional

import requests
from rapidfuzz import fuzz, process

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REVOLUT_LIST_FILE = DATA_DIR / "revolut_list.txt"
CSV_OUT = DATA_DIR / "revolut_mapping.csv"
JSON_OUT = DATA_DIR / "revolut_mapping.json"

UA = {"User-Agent": "revolut-mapper/1.0 (+github actions)"}

# ---------- HTTP ----------
def get_json(url: str, retries: int = 4, timeout: int = 30):
    backoff = 1.6
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff ** (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(backoff ** (i + 1))
    return None

# ---------- Parse Revolut list (from your file) ----------
def parse_revolut(raw_text: str) -> List[Dict[str, str]]:
    # Input is alphabetized sections with commas; entries like "ADA (Cardano)" or "GMX"
    tokens = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or len(line) == 1:  # skip A/B/C headers
            continue
        tokens.extend([t.strip() for t in line.split(",") if t.strip()])
    items = []
    for token in tokens:
        m = re.match(r"^([A-Z0-9]+)\s*\((.+?)\)$", token)
        if m:
            sym, name = m.group(1).strip(), m.group(2).strip()
        else:
            sym = token.split()[0].strip()
            name = token if " " in token else sym
        sym = sym.upper()
        if sym in {"SYNTH"}:  # category label in the source list
            continue
        items.append({"symbol": sym, "name": name})
    # de-dupe by symbol
    out, seen = [], set()
    for it in items:
        if it["symbol"] in seen:
            continue
        seen.add(it["symbol"])
        out.append(it)
    return out

# ---------- Public APIs ----------
def fetch_coingecko_list() -> List[Dict]:
    return get_json("https://api.coingecko.com/api/v3/coins/list?include_platform=false")

def fetch_binance_pairs() -> List[Dict]:
    data = get_json("https://api.binance.com/api/v3/exchangeInfo")
    return data.get("symbols", []) if data else []

def fetch_coinbase_products() -> List[Dict]:
    return get_json("https://api.exchange.coinbase.com/products") or []

def fetch_kraken_pairs() -> Dict:
    data = get_json("https://api.kraken.com/0/public/AssetPairs")
    return data.get("result", {}) if data else {}

def fetch_bitstamp_pairs() -> List[Dict]:
    return get_json("https://www.bitstamp.net/api/v2/trading-pairs-info/") or []

# ---------- Matching helpers ----------
def best_coingecko_match(symbol: str, name: str, cg_list: List[Dict]) -> Optional[str]:
    sym_lc = symbol.lower()
    exact = [c for c in cg_list if c.get("symbol","").lower() == sym_lc]
    if len(exact) == 1:
        return exact[0]["id"]
    if len(exact) > 1:
        choices = {c["id"]: c["name"] for c in exact}
        best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
        return best[0] if best and best[1] >= 70 else exact[0]["id"]
    # Fuzzy by name
    choices = {c["id"]: c["name"] for c in cg_list}
    best = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    return best[0] if best and best[1] >= 80 else None

def collect_binance_pairs(sym: str, info: List[Dict]) -> List[str]:
    out = []
    for s in info:
        if s.get("status") != "TRADING":
            continue
        base, quote = s.get("baseAsset"), s.get("quoteAsset")
        if base == sym and quote in {"USDT","FDUSD","USDC","BUSD","USD","EUR","TRY","BTC","ETH"}:
            out.append(f"{base}{quote}")
    return sorted(set(out))

def collect_coinbase_pairs(sym: str, prods: List[Dict]) -> List[str]:
    out = []
    for p in prods:
        if p.get("status") != "online":
            continue
        base = p.get("base_currency") or (p.get("id","").split("-")[0] if "-" in p.get("id","") else "")
        quote = p.get("quote_currency") or (p.get("id","").split("-")[1] if "-" in p.get("id","") else "")
        if base.upper() == sym and quote.upper() in {"USD","USDT","USDC","EUR","GBP","BTC","ETH"}:
            out.append(f"{base.upper()}-{quote.upper()}")
    return sorted(set(out))

def kraken_norm(x: str) -> str:
    # Kraken prefixes: XXBT, XETH, ZUSD ...
    return re.sub(r"^[XZ]", "", x or "").upper()

def collect_kraken_pairs(sym: str, pairs: Dict) -> List[str]:
    out = []
    for k,v in pairs.items():
        base, quote = kraken_norm(v.get("base")), kraken_norm(v.get("quote"))
        if base == sym and quote in {"USD","USDT","EUR","USDC","BTC","ETH"}:
            out.append(k)
    return sorted(set(out))

def collect_bitstamp_pairs(sym: str, pairs: List[Dict]) -> List[str]:
    out = []
    for p in pairs:
        nm = (p.get("name","") or "").upper()  # usually like "btcusd"
        if nm.startswith(sym):
            quote = nm[len(sym):]
            if quote in {"USD","USDT","EUR","BTC","ETH"}:
                out.append(f"{sym}{quote}")
    return sorted(set(out))

# ---------- Main ----------
def main():
    if not REVOLUT_LIST_FILE.exists():
        raise SystemExit(f"Missing list file: {REVOLUT_LIST_FILE}")

    raw = REVOLUT_LIST_FILE.read_text(encoding="utf-8")
    revolut_coins = parse_revolut(raw)
    print(f"Loaded Revolut symbols: {len(revolut_coins)}")

    cg = fetch_coingecko_list()
    bn = fetch_binance_pairs()
    cb = fetch_coinbase_products()
    kr = fetch_kraken_pairs()
    bs = fetch_bitstamp_pairs()

    rows = []
    for it in revolut_coins:
        sym, name = it["symbol"], it["name"]
        try:
            cg_id = best_coingecko_match(sym, name, cg) or ""
            bn_pairs = collect_binance_pairs(sym, bn)
            cb_pairs = collect_coinbase_pairs(sym, cb)
            kr_pairs = collect_kraken_pairs(sym, kr)
            bs_pairs = collect_bitstamp_pairs(sym, bs)
            rows.append({
                "symbol": sym,
                "name": name,
                "coingecko_id": cg_id,
                "binance_pairs": ";".join(bn_pairs),
                "coinbase_pairs": ";".join(cb_pairs),
                "kraken_pairs": ";".join(kr_pairs),
                "bitstamp_pairs": ";".join(bs_pairs),
            })
        except Exception as e:
            rows.append({
                "symbol": sym,
                "name": name,
                "coingecko_id": "",
                "binance_pairs": "",
                "coinbase_pairs": "",
                "kraken_pairs": "",
                "bitstamp_pairs": "",
                "error": str(e)[:200],
            })

    # CSV
    fields = ["symbol","name","coingecko_id","binance_pairs","coinbase_pairs","kraken_pairs","bitstamp_pairs"]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in fields})

    # JSON
    JSON_OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(rows)} rows →")
    print(f"  - {CSV_OUT}")
    print(f"  - {JSON_OUT}")

if __name__ == "__main__":
    main()
