import json, os, time

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

snap  = load_json("artifacts/market_snapshot.json")
sigs  = load_json("data/signals.json")
stats = load_json("artifacts/run_stats.json")

out = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime()),
    "regime": snap.get("regime"),
    "equity": snap.get("equity"),
    "cash": snap.get("cash"),
    "candidates": snap.get("candidates", []),
    "signals": sigs,
    "run_stats": stats,
}

os.makedirs("public_runs/latest", exist_ok=True)
with open("public_runs/latest/summary.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)