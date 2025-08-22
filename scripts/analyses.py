
# scripts/analyses.py

try:
    # when run as package
    from .marketdata import get_btc_5m_klines, ema, vwap
except ImportError:
    # fallback if relative fails
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts.marketdata import get_btc_5m_klines, ema, vwap

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True, parents=True)
SNAP_PATH = ARTIFACTS / "market_snapshot.json"

# ---------- helpers for HTTP ----------
import urllib.request

def _j(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "rev-scan/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

# ---------- exchange adapters (lightweight) ----------

def binance_24h_and_book(symbol_usdt: str):
    """
    Returns dict: {'ok': bool, 'price': float, 'volume_usd': float, 'bid': float, 'ask': float}
    symbol_usdt like 'ARBUSDT'
    """
    try:
        t = _j(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol_usdt}")
        last = float(t["lastPrice"])
        vol_base = float(t["volume"])
        vol_usd = vol_base * last
        ob = _j(f"https://api.binance.com/api/v3/depth?symbol={symbol_usdt}&limit=5")
        bid = float(ob["bids"][0][0]); ask = float(ob["asks"][0][0])
        return {"ok": True, "price": last, "volume_usd": vol_usd, "bid": bid, "ask": ask}
    except Exception:
        return {"ok": False}

def binance_klines_1m(symbol_usdt: str, limit: int = 120):
    """
    Returns list of dicts: [{'ts':ms,'o':...,'h':...,'l':...,'c':...,'v':...}]
    """
    urls = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
        f"https://api3.binance.com/api/v3/klines?symbol={symbol_usdt}&interval=1m&limit={limit}",
    ]
    for u in urls:
        try:
            arr = _j(u)
            return [
                {'ts': x[0], 'o': float(x[1]), 'h': float(x[2]), 'l': float(x[3]),
                 'c': float(x[4]), 'v': float(x[5])}
                for x in arr
            ]
        except Exception:
            continue
    return []

# ---------- TA utils ----------

def median(seq: List[float]) -> float:
    return statistics.median(seq) if seq else 0.0

def true_range(h, l, pc):
    return max(h-l, abs(h-pc), abs(l-pc))

def atr_from_klines(bars: List[Dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(1, period+1):
        h = bars[-i]['h']; l = bars[-i]['l']; pc = bars[-i-1]['c']
        trs.append(true_range(h, l, pc))
    return sum(trs) / len(trs)

# ---------- rule checks ----------

def check_regime() -> Dict:
    bars = get_btc_5m_klines(limit=120)
    if not bars:
        return {'ok': False, 'reason': 'no-candles-all-sources'}
    closes = [b['c'] for b in bars]
    ema9 = ema(closes, period=9)
    vw = vwap(bars)
    last = closes[-1]
    ok = (last > vw) and (last > ema9)
    return {'ok': ok, 'reason': '' if ok else 'close<=vwap/ema',
            'last_close': last, 'vwap': vw, 'ema9': ema9, 'source': bars[-1].get('src','')}

def spread_pct(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return 0.0 if mid == 0 else (ask - bid) / mid

def aggressive_breakout(bars_1m: List[Dict]) -> Optional[Dict]:
    """
    Rule 3: 1-min price +1.8% to +4.0% vs prev close, RVOL >= 4.0 vs last 15-min median,
            and break above last 15-min high with follow-through.
    """
    if len(bars_1m) < 20:
        return None
    last = bars_1m[-1]; prev = bars_1m[-2]
    pct = (last['c'] / prev['c']) - 1.0
    if not (0.018 <= pct <= 0.04):
        return None
    vol_med = median([b['v'] for b in bars_1m[-16:-1]])
    rvol = (last['v'] / vol_med) if vol_med > 0 else 0.0
    if rvol < 4.0:
        return None
    hh15 = max(b['h'] for b in bars_1m[-15:])
    if last['c'] <= hh15 * 1.0005:  # need a tiny close > 15m high for follow-through
        return None
    return {'pct': pct, 'rvol': rvol, 'hh15': hh15, 'last': last}

def micro_pullback_ok(bars_1m: List[Dict]) -> Optional[Dict]:
    """
    Rule 4: After breakout, a 15–45 sec pullback ≤ 0.6% from high and hold.
    Without tick feed, approximate by requiring:
      - last 1m bar's (high - close) / high <= 0.006  (no deep fade)
      - last 1m low is within 0.6% of high (micro dip only)
    We use last bar as the 'entry context'; stop = that bar’s low.
    """
    if len(bars_1m) < 2:
        return None
    last = bars_1m[-1]
    if last['h'] == 0 or last['c'] == 0:
        return None
    fade = (last['h'] - last['c']) / last['h']
    micro = (last['h'] - last['l']) / last['h']
    if fade <= 0.006 and micro <= 0.006:
        return {'entry': last['c'], 'pullback_low': last['l']}
    return None

# ---------- universe & mapping ----------

def load_revolut_mapping() -> List[Dict]:
    with open("data/revolut_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)

def best_binance_symbol(entry: Dict) -> Optional[str]:
    # prefer explicit binance symbol; else try <TICKER>USDT common form
    if entry.get("binance"):
        return entry["binance"]
    t = entry.get("ticker", "").upper()
    if not t:
        return None
    # special cases could be handled here if needed
    return f"{t}USDT"

# ---------- position sizing ----------

def position_size(entry: float, stop: float, equity: float, cash: float,
                  strong_regime: bool) -> float:
    """
    Returns USD buy amount respecting risk (1.2%), allocation cap (60/30%), cash, and nonzero distance.
    """
    risk_dollars = equity * 0.012
    dist = max(entry - stop, 0.0000001)
    qty = risk_dollars / dist
    usd = qty * entry

    alloc_cap = 0.60 if strong_regime else 0.30
    max_alloc = equity * alloc_cap
    usd = min(usd, max_alloc, cash)
    return max(0.0, usd)

# ---------- MAIN ----------

def main():
    equity = float(os.getenv("EQUITY", "41000"))
    cash   = float(os.getenv("CASH", "32000"))
    floor  = 36500.0
    buffer_ok = (equity - floor) >= 1000.0

    snapshot = {
        "time": _now_iso(),
        "equity": equity,
        "cash": cash,
        "breach": False,
        "breach_reason": "",
        "regime": {},
        "universe_count": 0,
        "candidates": []
    }

    # Floor / preservation checks first
    if not buffer_ok or equity <= 37500.0:
        snapshot["breach"] = True
        if not buffer_ok:
            snapshot["breach_reason"] = "buffer<1000_over_floor"
        elif equity <= 37500.0:
            snapshot["breach_reason"] = "equity<=37500"
        ARTIFACTS.mkdir(exist_ok=True, parents=True)
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        # Output A
        print("Raise cash now: exit weakest non-ETH, non-DOT positions on support breaks; halt new entries.")
        return

    # 1) Regime
    regime = check_regime()
    snapshot["regime"] = regime

    strong_regime = bool(regime.get("ok", False))

    # 2) Universe: Revolut coins passing volume/spread, excluding ETH & DOT from rotation
    mapping = load_revolut_mapping()
    universe = []
    for m in mapping:
        tkr = m.get("ticker","").upper()
        if tkr in ("ETH","DOT"):
            continue  # excluded from rotation
        sym = best_binance_symbol(m)
        if not sym:
            continue

        md = binance_24h_and_book(sym)
        if not md["ok"]:
            continue
        spr = spread_pct(md["bid"], md["ask"])
        if md["volume_usd"] >= 8_000_000 and spr <= 0.005:
            universe.append({
                "ticker": tkr,
                "binance": sym,
                "price": md["price"],
                "bid": md["bid"],
                "ask": md["ask"],
                "spread": spr,
                "vol_usd": md["volume_usd"]
            })

    snapshot["universe_count"] = len(universe)

    # 3 & 4) Signals
    scored = []
    for u in universe:
        bars = binance_klines_1m(u["binance"], limit=120)
        if not bars:
            continue
        br = aggressive_breakout(bars)
        if not br:
            continue
        mp = micro_pullback_ok(bars)
        if not mp:
            continue
        # ATR(1m)
        atr1m = atr_from_klines(bars, period=14)
        if atr1m <= 0:
            continue

        # Score: prioritize higher rvol and pct (simple linear score)
        score = br["rvol"] * (1 + br["pct"])
        candidate = {
            "ticker": u["ticker"],
            "symbol": u["binance"],
            "entry": mp["entry"],
            "pullback_low": mp["pullback_low"],
            "atr1m": atr1m,
            "rvol": br["rvol"],
            "pct": br["pct"],
            "score": score
        }
        scored.append(candidate)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # If regime weak, only top 1 is considered anyway (handled by sizing rule)
    if not scored:
        # Output C
        snapshot["candidates"] = []
        SNAP_PATH.write_text(json.dumps(snapshot, indent=2))
        print("Hold and wait.")
        return

    top = scored[0]  # take only the single best candidate
    entry = top["entry"]
    stop  = top["pullback_low"]
    atr   = top["atr1m"]

    # Targets
    t1 = entry + 0.8 * atr
    t2 = entry + 1.5 * atr

    buy_usd = position_size(entry, stop, equity, cash, strong_regime)

    snapshot["candidates"] = [top]
    SNAP_PATH.write_text(json.dumps(snapshot, indent=2))

    # Output B in the exact required bullet format
    # • Coin name + ticker
    # • Entry price
    # • Target
    # • Stop-loss / exit plan
    # • What to sell from portfolio (excluding ETH, DOT)
    # • Exact USD buy amount so total equity ≥ $36,500

    # We don't have your live portfolio composition here; follow rotation rule:
    sell_plan = "Sell weakest non-ETH, non-DOT on support break if cash needed."

    lines = [
        f"• {top['ticker']} + {top['ticker']}",
        f"• Entry price: {entry:.6f}",
        f"• Target: T1 {t1:.6f}, T2 {t2:.6f}, trail after +1.0R",
        f"• Stop-loss / exit plan: Invalidate below micro-pullback low {stop:.6f} or stall > 5 min",
        f"• What to sell from portfolio (excluding ETH, DOT): {sell_plan}",
        f"• Exact USD buy amount so total equity ≥ $36,500: ${buy_usd:,.2f}"
    ]
    print("\n".join(lines))

if __name__ == "__main__":
    main()
