# revolut-crypto-mapping

Builds a **complete mapping** of all Revolut-listed cryptocurrencies to:
- CoinGecko IDs (for metadata/price)
- Binance / Coinbase / Kraken / Bitstamp tickers (for open-source live market data)

## Outputs (auto-generated)
- `data/revolut_mapping.csv`
- `data/revolut_mapping.json`

## How it works
- Master list of Revolut coins is kept in `data/revolut_list.txt` (from your provided list).
- The script queries public APIs (no keys needed):
  - CoinGecko `/coins/list`
  - Binance `/api/v3/exchangeInfo`
  - Coinbase `/products`
  - Kraken `/0/public/AssetPairs`
  - Bitstamp `/api/v2/trading-pairs-info/`
- It matches each Revolut symbol to CoinGecko + exchange tickers.

## Run locally
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/map_revolut_coins.py
