#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Revolut ↔ Binance mapping.
"""

import json, sys, time, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REV_FILE = DATA_DIR / "revolut_list.json"
OUT_JSON = DATA_DIR / "revolut_mapping.json"
OUT_CSV  = DATA_DIR / "revolut_mapping.csv"

PREFERRED_QUOTES = ("USDT","USDC","FDUSD","BUSD","TUSD","USD")

BINANCE_URLS = (
    "https://data-api.binance.vision/api/v3/exchangeInfo",
    "https://api.binance.com/api/v3/exchangeInfo",
)

session = requests.Session()
session.headers.update({"User-Agent": "revolut-crypto-mapping"})

def log(msg): print(f"[mapping] {msg}", flush=True)

def get_json(urls):
    last_err=None
    for u in urls:
        for i in range(3):
            try:
                r=session.get(u,timeout=20)
                if r.status_code in (429,418,451):
                    time.sleep(2**i)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err=e
                time.sleep(2**i)
    raise last_err

def load_revolut():
    with REV_FILE.open() as f: data=json.load(f)
    if not isinstance(data,dict): raise ValueError("revolut_list.json must be dict")
    return data

def fetch_binance():
    data=get_json(BINANCE_URLS)
    syms=data.get("symbols",[])
    return [s for s in syms if s.get("status")=="TRADING"]

def choose(cands):
    if not cands: return None
    ranked=sorted(cands,key=lambda x:(PREFERRED_QUOTES.index(x[2]) if x[2] in PREFERRED_QUOTES else 99))
    return ranked[0]

def main():
    rev=load_revolut()
    bins=fetch_binance()
    by_base={}
    for s in bins:
        base,quote=s["baseAsset"].upper(),s["quoteAsset"].upper()
        sym=s["symbol"]
        by_base.setdefault(base,[]).append((sym,base,quote))

    rows=[]
    for r_ticker,r_name in rev.items():
        cands=by_base.get(r_ticker.upper(),[])
        ch=choose(cands)
        if not ch: continue
        sym,_,_=ch
        rows.append({
            "revolut_ticker":r_ticker,
            "binance_symbol":sym
        })

    rows=sorted(rows,key=lambda d:d["revolut_ticker"])
    log(f"mapped {len(rows)} / {len(rev)}")

    DATA_DIR.mkdir(exist_ok=True)
    with OUT_JSON.open("w") as f: json.dump(rows,f,indent=2)
    with OUT_CSV.open("w") as f:
        f.write("revolut_ticker,binance_symbol\n")
        for r in rows: f.write(f"{r['revolut_ticker']},{r['binance_symbol']}\n")
    return 0

if __name__=="__main__": sys.exit(main())