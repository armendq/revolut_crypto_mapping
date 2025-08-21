import os
import csv
import json

# --- Full Revolut coins mapping (symbol -> display name) ---
REVOLUT_COINS = {
    # A
    "1INCH": "1INCH",
    "AAVE": "Aave",
    "ACA": "Acala Token",
    "ADA": "Cardano",
    "AERGO": "Aergo",
    "AGLD": "Adventure Gold",
    "AIOZ": "AIOZ Network",
    "AKT": "Akash Network",
    "ALGO": "Algorand",
    "ALICE": "MyNeighborAlice",
    "ALPHA": "Alpha Venture DAO",
    "AMP": "AMP",
    "ANKR": "Ankr",
    "APE": "ApeCoin",
    "API3": "API3",
    "APT": "Aptos",
    "AR": "Arweave",
    "ARB": "Arbitrum",
    "ARKM": "Arkham",
    "ARPA": "ARPA",
    "ASTR": "Astar",
    "ATOM": "Cosmos",
    "AUDIO": "Audius",
    "AVAX": "Avalanche",
    "AXS": "Axie Infinity",

    # B
    "BAL": "Balancer",
    "BAND": "Band Protocol",
    "BAT": "Basic Attention Token",
    "BCH": "Bitcoin Cash",
    "BICO": "Biconomy",
    "BLUR": "Blur",
    "BLZ": "Bluzelle",
    "BNC": "Bifrost",
    "BNT": "Bancor",
    "BONK": "Bonk",
    "BOND": "BarnBridge",
    "BTC": "Bitcoin",
    "BTG": "Bitcoin Gold",
    "BTRST": "Braintrust",
    "BTT": "BitTorrent",

    # C
    "CELO": "Celo",
    "CELR": "Celer Network",
    "CFX": "Conflux",
    "CHZ": "Chiliz",
    "CHR": "Chromia",
    "CLV": "Clover Finance",
    "COMP": "Compound",
    "COTI": "COTI",
    "CQT": "Covalent",
    "CRO": "Cronos",
    "CRV": "Curve DAO Token",
    "CTSI": "Cartesi",
    "CVC": "Civic",
    "CVX": "Convex Finance",

    # D
    "DAI": "Dai",
    "DAR": "Mines of Dalarnia",
    "DASH": "Dash",
    "DENT": "Dent",
    "DGB": "DigiByte",
    "DIMO": "DIMO",
    "DNT": "district0x",
    "DOGE": "Dogecoin",
    "DOT": "Polkadot",
    "DREP": "DREP",
    "DYDX": "dYdX",

    # E
    "EGLD": "MultiversX",
    "ENA": "Ethena",
    "ENJ": "Enjin Coin",
    "ENS": "Ethereum Name Service",
    "EOS": "EOS",
    "ETC": "Ethereum Classic",
    "ETH": "Ethereum",
    "ETHW": "EthereumPoW",
    "EUL": "Euler",

    # F
    "FET": "Fetch.ai",
    "FIDA": "Bonfida",
    "FIL": "Filecoin",
    "FLOKI": "FLOKI",
    "FLOW": "Flow",
    "FLR": "Flare",
    "FORTH": "Ampleforth Governance Token",
    "FTM": "Fantom",
    "FXS": "Frax Share",

    # G
    "GALA": "Gala",
    "GFI": "Goldfinch",
    "GHST": "Aavegotchi",
    "GLM": "Golem",
    "GLMR": "Moonbeam",
    "GMT": "STEPN",
    "GMX": "GMX",
    "GNO": "Gnosis",
    "GNS": "Gains Network",
    "GODS": "Gods Unchained",
    "GRT": "The Graph",

    # H
    "HBAR": "Hedera",
    "HFT": "Hashflow",
    "HIGH": "Highstreet",
    "HOPR": "HOPR",
    "HOT": "Holo",

    # I
    "ICP": "Internet Computer",
    "IDEX": "IDEX",
    "ILV": "Illuvium",
    "IMX": "Immutable",
    "INJ": "Injective",
    "IOTX": "IoTeX",

    # J
    "JASMY": "JasmyCoin",
    "JTO": "Jito",
    "JUP": "Jupiter",

    # K
    "KAVA": "Kava",
    "KDA": "Kadena",
    "KNC": "Kyber Network",
    "KSM": "Kusama",

    # L
    "LDO": "Lido DAO",
    "LINK": "Chainlink",
    "LPT": "Livepeer",
    "LQTY": "Liquity",
    "LRC": "Loopring",
    "LSK": "Lisk",
    "LTC": "Litecoin",

    # M
    "MAGIC": "MAGIC",
    "MANA": "Decentraland",
    "MASK": "Mask Network",
    "MATIC": "Polygon",
    "MC": "Merit Circle",
    "MDT": "Measurable Data Token",
    "MINA": "Mina",
    "MKR": "Maker",
    "MLN": "Enzyme",
    "MPL": "Maple",
    "MTRG": "Meter Governance",

    # N
    "NEAR": "NEAR Protocol",
    "NEXO": "Nexo",
    "NKN": "NKN",

    # O
    "OCEAN": "Ocean Protocol",
    "OGN": "Origin Protocol",
    "OMG": "OMG Network",
    "ONDO": "Ondo",
    "ONG": "Ontology Gas",
    "ONT": "Ontology",
    "OP": "Optimism",
    "ORCA": "Orca",
    "OXT": "Orchid",

    # P
    "PAXG": "PAX Gold",
    "PENDLE": "Pendle",
    "PEPE": "PEPE",
    "PERP": "Perpetual Protocol",
    "PIXEL": "Pixels",
    "PLA": "PlayDapp",
    "POKT": "Pocket Network",
    "POLS": "Polkastarter",
    "POLYX": "Polymesh",
    "POND": "Marlin",
    "POWR": "Powerledger",
    "PUNDIX": "Pundi X",

    # Q
    "QI": "BENQI",
    "QNT": "Quant",
    "QTUM": "Qtum",

    # R
    "RAD": "Radicle",
    "RARE": "SuperRare",
    "RARI": "Rarible",
    "RAY": "Raydium",
    "REN": "Ren",
    "REQ": "Request",
    "RLC": "iExec RLC",
    "RLY": "Rally",
    "RNDR": "Render",
    "RON": "Ronin",
    "ROSE": "Oasis Network",
    "RPL": "Rocket Pool",
    "RSR": "Reserve Rights",

    # S
    "SAND": "The Sandbox",
    "SC": "Siacoin",
    "SGB": "Songbird",
    "SHIB": "Shiba Inu",
    "SKL": "SKALE Network",
    "SLP": "Smooth Love Potion",
    "SNT": "Status",
    "SNX": "Synthetix",
    "SOL": "Solana",
    "SPELL": "Spell Token",
    "STG": "Stargate Finance",
    "STORJ": "Storj",
    "STRK": "Starknet",
    "STX": "Stacks",
    "SUI": "Sui",
    "SUSHI": "SushiSwap",
    "SYN": "Synapse",
    "SYNTH": "SYNTH",

    # T
    "T": "Threshold",
    "TEL": "Telcoin",
    "TIA": "Celestia",
    "TON": "Toncoin",
    "TRB": "Tellor",
    "TRU": "TrueFi",
    "TRX": "TRON",

    # U
    "UMA": "UMA",
    "UNI": "Uniswap",
    "USDC": "USD Coin",
    "USDT": "Tether",

    # V
    "VET": "VeChain",
    "VOXEL": "Voxies",
    "VTHO": "VeThor Token",

    # W
    "WAVES": "Waves",
    "WAXP": "WAX",
    "WBTC": "Wrapped Bitcoin",
    "WCFG": "Wrapped Centrifuge",
    "WLD": "Worldcoin",
    "WOO": "WOO Network",

    # X
    "XLM": "Stellar",
    "XMR": "Monero",
    "XRP": "XRP",
    "XTZ": "Tezos",
    "XVS": "Venus",

    # Y
    "YFI": "yearn.finance",
    "YGG": "Yield Guild Games",

    # Z
    "ZEC": "Zcash",
    "ZETA": "ZetaChain",
    "ZIL": "Zilliqa",
    "ZRX": "0x"
}

# --- Generate outputs ---
os.makedirs("data", exist_ok=True)

# CSV
with open("data/revolut_mapping.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["symbol", "name"])
    writer.writeheader()
    for symbol, name in sorted(REVOLUT_COINS.items()):
        writer.writerow({"symbol": symbol, "name": name})

# JSON
with open("data/revolut_mapping.json", "w", encoding="utf-8") as f:
    json.dump(REVOLUT_COINS, f, indent=2, ensure_ascii=False)

print("✅ Generated data/revolut_mapping.csv and data/revolut_mapping.json")