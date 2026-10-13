# Crypto address scanner

Open `/crypto-scanner` from Home → Explore → Research or the tool navigation.
The gateway exposes `GET /api/crypto/scan?address=0x...&days=30&limit=100`.
Supported windows are 7, 30, and 90 days; record caps are 100 and 250 per feed.
The server needs outbound HTTPS access to `eth.blockscout.com`; no API key is required.

The scanner reads normal transactions, internal calls, token transfers, and current
ETH balance from the public Blockscout account API. It does not save addresses or
results to the application database. Standard server/provider logs may record requests.
JSON exports include scan time, coverage, normalized activity, and the contract registry.

The Ethereum-only registry contains 19 classic pools and the router in the official
Tornado Cash CLI configuration, checked 2026-10-12:
https://github.com/tornadocash/tornado-cli/blob/master/config.js
It intentionally excludes underlying asset contracts and generic multicall contracts.
It is not an exhaustive historic deployment or cross-chain registry.

Matching compares the direct counterparty to the registry. It distinguishes failed
attempts, successful/recorded activity, and unknown status. A scanned pool itself is
identified separately. Incoming transactions do not establish intent; internal call
values may be call context rather than transfers. The tool does not trace indirect
funding, identify owners, determine sanctions status, or assign a risk score.
Multiple feed records can refer to one transaction. Explorer indexing, unavailable
feeds, and record caps prevent a negative result from proving no historical contact.

Validation: `.venv-integration/Scripts/python.exe -m unittest test_crypto_scanner`
uses deterministic fixtures for evidence classification, coverage, validation, exact
decimal formatting, browser filters, themes, exports, and mobile layout.
