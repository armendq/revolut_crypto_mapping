import shutil
from pathlib import Path

PUBLIC = Path("public_runs")
LATEST = PUBLIC / "latest"

# make sure latest folder exists
LATEST.mkdir(parents=True, exist_ok=True)

# list of files we want to copy into "latest"
files = ["signals.json", "summary.json", "debug_scan.json", "market_snapshot.json", "run_stats.json"]

for f in files:
    src = PUBLIC / f
    if src.exists():
        shutil.copy(src, LATEST / f)
        print(f"Copied {src} → {LATEST/f}")
    else:
        print(f"⚠️ Missing {src}, skipped.")