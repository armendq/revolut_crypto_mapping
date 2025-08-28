name: Build & Publish Public Runs (complete)

on:
  workflow_dispatch:
  schedule:
    - cron: "*/15 * * * *"
  push:
    branches: [ main ]
    paths:
      - analyses.py
      - scripts/**
      - public_runs/**
      - .github/workflows/build.yaml

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  TZ: Europe/Prague
  DEFAULT_BRANCH: main
  SUMMARY_URL: https://raw.githubusercontent.com/armendq/revolut_crypto_mapping/main/public_runs/latest/summary.json
  SITE_DIR: site
  RUN_DIR: public_runs/latest
  REPO_URL: https://github.com/${{ github.repository }}
  PAGES_URL: https://${{ github.repository_owner }}.github.io/${{ github.event.repository.name }}

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 25

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ env.DEFAULT_BRANCH }}

      - name: Configure Git user
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Cache pip
        id: cache-pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          set -e
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          else
            pip install pandas numpy requests python-dateutil pytz jq
          fi

      - name: Ensure folders
        run: |
          mkdir -p "${RUN_DIR}"
          mkdir -p "${SITE_DIR}"

      # ---------------- Fetch with exactly 2 attempts (60s wait) ----------------
      - name: Fetch latest summary (2 attempts, 60s wait)
        id: fetch
        shell: bash
        run: |
          set -euo pipefail
          TMP="${RUN_DIR}/summary.tmp.json"
          OUT="${RUN_DIR}/summary.json"
          FALLBACK="${RUN_DIR}/last_good.json"
          STATUS="${SITE_DIR}/fetch_status.txt"

          attempt() {
            echo "::group::curl"
            HTTP=$(curl -sS -w "%{http_code}" -H "Cache-Control: no-cache" -o "$TMP" "${SUMMARY_URL}" || echo "000")
            echo "HTTP=$HTTP"
            echo "::endgroup::"
            if [ "$HTTP" != "200" ]; then
              return 1
            fi
            if ! jq -e . < "$TMP" >/dev/null 2>&1; then
              return 1
            fi
            mv "$TMP" "$OUT"
            return 0
          }

          echo "Attempt 1..."
          if attempt; then
            echo "ok=true" >> "$GITHUB_OUTPUT"
            echo "Fetch OK on attempt 1" | tee "$STATUS"
            exit 0
          fi

          echo "Waiting 60 seconds before the second attempt..."
          sleep 60

          echo "Attempt 2..."
          if attempt; then
            echo "ok=true" >> "$GITHUB_OUTPUT"
            echo "Fetch OK on attempt 2" | tee "$STATUS"
            exit 0
          fi

          echo "Hold and wait. Fetch failed twice." | tee "$STATUS"
          # fallback to last_good if exists and valid; still treat as not-ok for downstream guards
          if [ -f "$FALLBACK" ] && jq -e . "$FALLBACK" >/dev/null 2>&1; then
            cp "$FALLBACK" "$OUT"
            echo "Used last_good.json fallback" | tee -a "$STATUS"
          fi
          echo "ok=false" >> "$GITHUB_OUTPUT"

      # ---------------- Basic schema checks (non-fatal) ----------------
      - name: Validate summary schema (light)
        if: always()
        shell: bash
        run: |
          SRC="${RUN_DIR}/summary.json"
          REPORT="${SITE_DIR}/validation.json"
          if [ ! -f "$SRC" ]; then
            jq -n '{ok:false, reason:"summary.json missing"}' > "$REPORT"
            exit 0
          fi
          # Require top-level object
          if ! jq -e 'type=="object"' "$SRC" >/dev/null; then
            jq -n '{ok:false, reason:"top-level is not an object"}' > "$REPORT"
            exit 0
          fi
          # Optional keys sanity (do not fail build)
          SIG=$(jq -r '.signals.type // empty' "$SRC")
          jq -n --arg sig "$SIG" '{ok:true, signals_type:$sig}' > "$REPORT"

      # ---------------- Revolut coverage check & CRO guard ----------------
      - name: Build Revolut universe list
        shell: bash
        run: |
          cat > revolut_universe.json <<'JSON'
          { "CRO":"Cronos", "BTC":"Bitcoin", "ETH":"Ethereum", "DOT":"Polkadot", "LPT":"Livepeer",
            "SOL":"Solana","ADA":"Cardano","AVAX":"Avalanche","MATIC":"Polygon","ATOM":"Cosmos",
            "LINK":"Chainlink","DOGE":"Dogecoin","XRP":"XRP","LTC":"Litecoin","BNB":"BNB",
            "RNDR":"Render","INJ":"Injective","APT":"Aptos","ARB":"Arbitrum","OP":"Optimism",
            "NEAR":"NEAR Protocol","SUI":"Sui","TIA":"Celestia","JUP":"Jupiter","ONDO":"Ondo",
            "PEPE":"PEPE","DIMO":"DIMO","QNT":"Quant","AAVE":"Aave","GRT":"The Graph","FTM":"Fantom",
            "ETC":"Ethereum Classic","BCH":"Bitcoin Cash","XLM":"Stellar","XTZ":"Tezos","EGLD":"MultiversX"
          }
          JSON

      - name: Coverage & missing tickers report
        shell: bash
        run: |
          set -euo pipefail
          SRC="${RUN_DIR}/summary.json"
          UNIV=revolut_universe.json
          REPORT="${SITE_DIR}/missing_tickers.json"
          touch "$REPORT"
          if [ ! -f "$SRC" ]; then
            jq -n '{missing:["ALL (no summary.json)"]}' > "$REPORT"
            exit 0
          fi

          # Gather tickers mentioned in summary
          jq -r '
            [
              (.universe? // [])[],
              (.signals?.tickers? // []),
              (.candidates?[]?.ticker // empty)
            ] | map(tostring | ascii_upcase) | unique | .[]
          ' "$SRC" 2>/dev/null | sort -u > /tmp/seen.txt || true

          jq -r 'keys[]' "$UNIV" | sort -u > /tmp/rev.txt

          comm -23 /tmp/rev.txt /tmp/seen.txt > /tmp/miss.txt || true

          jq -n --argfile miss /tmp/miss.txt '
            { missing: ($miss|[.[]]|map(rtrimstr("\n"))|map(select(length>0))) }
          ' > "$REPORT"

          echo "First 20 missing (if any):"
          head -n 20 /tmp/miss.txt || true

      # ---------------- Optional: run analyses.py if present ----------------
      - name: Run analyses.py (if present)
        if: ${{ hashFiles('analyses.py') != '' }}
        shell: bash
        run: |
          set -euo pipefail
          echo "Running analyses.py..."
          python analyses.py || echo "analyses.py returned non-zero; continuing"

      # ---------------- Promote snapshot to last_good when fetch ok ----------------
      - name: Promote to last_good
        if: steps.fetch.outputs.ok == 'true'
        shell: bash
        run: |
          cp "${RUN_DIR}/summary.json" "${RUN_DIR}/last_good.json" || true

      # ---------------- Prepare site bundle ----------------
      - name: Prepare site content
        shell: bash
        run: |
          cp -f "${RUN_DIR}/summary.json"     "${SITE_DIR}/summary.json"     2>/dev/null || true
          cp -f "${RUN_DIR}/last_good.json"   "${SITE_DIR}/last_good.json"   2>/dev/null || true
          cp -f "${SITE_DIR}/missing_tickers.json" "${SITE_DIR}/missing_tickers.json" 2>/dev/null || true
          echo "status=$( [ -f ${RUN_DIR}/summary.json ] && echo ok || echo failed )" > "${SITE_DIR}/status.txt"

          cat > "${SITE_DIR}/index.html" <<'HTML'
          <!doctype html><html><head><meta charset="utf-8"><title>Public Runs</title>
          <meta name="viewport" content="width=device-width,initial-scale=1" />
          <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:24px;line-height:1.4}</style>
          </head><body>
            <h1>Public Runs</h1>
            <ul>
              <li><a href="summary.json">summary.json</a></li>
              <li><a href="last_good.json">last_good.json</a></li>
              <li><a href="missing_tickers.json">missing_tickers.json</a></li>
              <li><a href="validation.json">validation.json</a></li>
              <li><a href="fetch_status.txt">fetch_status.txt</a></li>
              <li><a href="status.txt">status.txt</a></li>
            </ul>
          </body></html>
          HTML

      # ---------------- Commit back to main ----------------
      - name: Commit & push updates
        shell: bash
        run: |
          set -euo pipefail
          git add "${RUN_DIR}"/*.json "${SITE_DIR}"/*
          if ! git diff --cached --quiet; then
            git commit -m "ci: update snapshots & site ($(date -u +'%Y-%m-%d %H:%M:%S') UTC) [skip ci]"
            git push origin "${DEFAULT_BRANCH}"
          else
            echo "No changes to commit."
          fi

      # ---------------- Configure & deploy GitHub Pages ----------------
      - name: Configure Pages
        uses: actions/configure-pages@v5

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ${{ env.SITE_DIR }}

      - name: Deploy to GitHub Pages
        id: pages
        uses: actions/deploy-pages@v4

      # ---------------- Verify Pages availability ----------------
      - name: Verify Pages (summary.json)
        shell: bash
        run: |
          set -e
          URL="${{ env.PAGES_URL }}/summary.json"
          echo "Verifying: $URL"
          for i in {1..10}; do
            if curl -fsS "$URL" >/dev/null; then
              echo "Pages OK"
              exit 0
            fi
            echo "Not ready yet (attempt $i). Sleeping 6s..."
            sleep 6
          done
          echo "Pages not reachable (non-fatal)."

      # ---------------- Upload artifacts for inspection ----------------
      - name: Upload run artifacts
        uses: actions/upload-artifact@v4
        with:
          name: public-runs-site
          path: |
            public_runs/latest/*.json
            site/*
          retention-days: 7

      # ---------------- Job Summary ----------------
      - name: Write job summary
        shell: bash
        run: |
          echo "## Build & Publish Summary" >> $GITHUB_STEP_SUMMARY
          echo "- Repo: ${{ github.repository }}" >> $GITHUB_STEP_SUMMARY
          echo "- Branch: ${{ env.DEFAULT_BRANCH }}" >> $GITHUB_STEP_SUMMARY
          echo "- Fetch OK: ${{ steps.fetch.outputs.ok }}" >> $GITHUB_STEP_SUMMARY
          echo "- Pages URL: ${{ env.PAGES_URL }}" >> $GITHUB_STEP_SUMMARY
          if [ -f site/missing_tickers.json ]; then
            echo "" >> $GITHUB_STEP_SUMMARY
            echo "### Missing tickers (first few)" >> $GITHUB_STEP_SUMMARY
            jq -r '.missing[:20] | map("- " + .)[]' site/missing_tickers.json >> $GITHUB_STEP_SUMMARY || true
          fi