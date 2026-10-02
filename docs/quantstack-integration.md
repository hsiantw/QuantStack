# Merged QuantStack workspace

The existing folder merge retains the Streamlit application in `QuantStack-main/`
and the collector, database APIs, and browser workspace at repository root.
`pages/market_workspace.py` connects those features to the Streamlit home screen
and navigation. No duplicate copies of either application are maintained.

## Render

Target: https://quantstack.onrender.com/

- Build command: `pip install -r requirements.txt`
- Start command: `python serve_quantstack.py --host 0.0.0.0`
- Port: Render's `PORT` environment variable, read by the launcher.

`render.yaml`, `Procfile`, `start.sh`, and the Replit configuration all use this
launcher. For a Render service configured manually, update its start command in
the service settings; changing `render.yaml` alone may not change those settings.
An existing `streamlit run QuantStack-main/app.py` service still gets the new
workspace page with the merged browser code and daily snapshot data.

The launcher keeps Streamlit and the read-only market API on loopback ports and
proxies them through the public port. `/workspace/` serves the chart workspace;
other routes serve Streamlit, including websocket sessions and uploads. Closing
the launcher stops both services. Market data endpoints contain public market
prices, not account or portfolio records.

## Data availability

The local market database is excluded from Git and is not uploaded by deployment.
When `data/market.sqlite` has stored prices, the launcher uses it for daily/hourly
charts, full indicators, screening, backtesting, and Markov analysis. Restart the
launcher after initially provisioning a database to switch out of snapshot mode.

Without that database, the current repository's browser workspace runs against
the daily snapshots at `https://hsiantw.github.io/market-atlas/`. The gateway
requests catalogs and per-symbol compressed histories on demand. It does not
forward account cookies or authentication headers to the dataset host. Snapshot
mode disables hourly bars, collector usage, and the full server indicator library.
The current chart and each asset's date identify data freshness; deployment does
not collect new market data.

For a different snapshot host, set `QUANTSTACK_SNAPSHOT_URL` to a URL with the
`symbols.json`, `screener.json`, and `prices/*.json.gz` structure produced by
`build_site.py`. Streamlit-only embedding also requires that host to permit CORS.

## Checks

Install `requirements.txt` and `playwright` into a working Python environment.
The browser checks use installed Microsoft Edge.

```powershell
python -m unittest test_quantstack_gateway -v
python serve_quantstack.py --port 8510
# In a separate terminal:
python smoke_quantstack.py --url http://127.0.0.1:8510
python smoke_workspace.py --url http://127.0.0.1:8510/workspace/
python smoke_strategy.py --url http://127.0.0.1:8510/workspace/
python smoke_markov.py --url http://127.0.0.1:8510/workspace/
python smoke_markov.py --static
# To test an existing Streamlit-only deployment:
python smoke_quantstack.py --url http://127.0.0.1:8511 --standalone
```

Gateway tests cover paths and queries, uploads, cookies, binary websocket messages,
subprotocols, origin validation, snapshot routing, and filesystem boundaries.
Browser checks exercise the actual Streamlit iframe and the existing research
workflows. `data/quantstack-integration.png` is a local screenshot from the
Streamlit integration check.
