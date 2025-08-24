# scripts/publish_latest.py
import json, shutil, datetime, os
from pathlib import Path

ART = Path("artifacts")
DATA = Path("data")
PUB  = Path("public_runs")
LATEST = PUB/"latest"
STAGE = PUB/"_staging"

def read_json(p): return json.loads(Path(p).read_text())

# ensure dirs
STAGE.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)

# build files in staging
summary = read_json(ART/"market_snapshot.json")
Path(STAGE/"summary.json").write_text(json.dumps(summary, indent=2))

signals_raw = read_json(DATA/"signals.json")
signals_wrapped = {
    "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "signals": signals_raw
}
Path(STAGE/"signals.json").write_text(json.dumps(signals_wrapped, indent=2))

for f in ("run_stats.json","debug_scan.json"):
    shutil.copy(ART/f, STAGE/f)

# atomically swap latest
if LATEST.exists():
    shutil.rmtree(LATEST)
STAGE.rename(LATEST)

# timestamped snapshot (keep newest only)
ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
snap = PUB/ts
snap.mkdir(parents=True, exist_ok=True)
for f in LATEST.iterdir():
    shutil.copy(f, snap/f.name)

# keep only newest timestamped folder
ts_dirs = sorted([d for d in PUB.iterdir() if d.is_dir() and d.name[:4].isdigit()], reverse=True)
for d in ts_dirs[1:]:
    shutil.rmtree(d)
print("Published to", LATEST)