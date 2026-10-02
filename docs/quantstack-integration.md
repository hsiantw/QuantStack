# Consolidated QuantStack workspace

The Streamlit application in `QuantStack-main/`, the collector, shared database,
and browser workspace form one project and one public site:
https://quantstack.onrender.com/.

Open **Market workspace** from the home screen or navigation for charts,
drawings, comparisons, screening, strategy backtests, and Markov analysis.
Existing portfolio, AI, risk, and statistical analysis pages remain available.

## Running and deploying

```powershell
python -m pip install -r requirements.txt
python prepare_snapshot.py
python serve_quantstack.py
```

Open http://127.0.0.1:8501. On Render, use `--host 0.0.0.0`; the launcher reads
`PORT`. See [hosting and publication](../HOSTING.md) for build and deployment
settings. The original Streamlit-only start command is also supported.

The launcher keeps Streamlit and the database API on loopback ports and proxies
them through one public port. `/workspace/` serves the browser workspace; other
routes serve Streamlit, including websocket sessions and uploads. Closing the
launcher stops both services. Public workspace endpoints contain market data,
not account or portfolio records.

## Data ownership and availability

QuantStack owns its `market-data` GitHub release and `market-snapshot.json`
manifest. `prepare_snapshot.py` verifies the checksum, restricts archive members
to the expected catalogs and histories, and installs a version atomically.
Browsers load catalogs and histories from the QuantStack app itself. The app has
no runtime dependency on the former standalone website.

The launcher uses `data/market.sqlite` when stored prices exist, supporting
native hourly charts and server indicators. Otherwise it serves its prepared
daily snapshot. Restart after initially provisioning a database to switch modes.
Snapshot mode disables hourly bars, collector usage, and the full indicator
library. Data timestamps remain visible; deployment is not a price refresh.

Streamlit-only mode uses the same prepared release through
`/app/static/market-data/<version>/`, with the current browser code embedded in
its page. Static serving is enabled by the checked-in configuration. Generated
files are excluded from Git. Browser-local favorites, drawings, and settings
keep their existing storage keys to avoid discarding saved work.

## Verification

Install `requirements.txt` and `playwright`. Browser checks use Microsoft Edge.

```powershell
python -m unittest test_quantstack_gateway test_snapshot -v
python serve_quantstack.py --port 8510
# In a separate terminal:
python smoke_quantstack.py --url http://127.0.0.1:8510
python smoke_workspace.py --url http://127.0.0.1:8510/workspace/
python smoke_strategy.py --url http://127.0.0.1:8510/workspace/
python smoke_markov.py --url http://127.0.0.1:8510/workspace/
python smoke_markov.py --static
# For a Streamlit-only server started on port 8511:
python smoke_quantstack.py --url http://127.0.0.1:8511 --standalone
```

Tests cover HTTP paths, uploads, cookies, binary websocket messages, subprotocols,
origin checks, local snapshot serving, archive validation, and actual browser
research workflows. The integration browser check saves a local screenshot at
`data/quantstack-integration.png`.
