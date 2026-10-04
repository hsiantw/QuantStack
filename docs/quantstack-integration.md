# Consolidated QuantStack workspace

The collector, database and native browser workspace form one project.
All supported analysis tools open within the chart workspace from the right rail.
See the [migration map](native-workspace-migration.md) for every former page.

## Running and deploying

```powershell
python -m pip install -r requirements.txt
python prepare_snapshot.py
python serve_quantstack.py
```

Open http://127.0.0.1:8501. On Render, use `--host 0.0.0.0`; the launcher reads
`PORT`. See [hosting and publication](../HOSTING.md) for build and deployment
settings. The former Streamlit launcher is retired.

The launcher keeps the database API on a loopback port and proxies it through
one public port. `/workspace/` serves the native browser workspace. The root,
including old `?research=1` links, redirects there. Old Streamlit routes, uploads
and websockets return 404. No Streamlit subprocess is started. Collector writes
are restricted to the local host and same-origin requests. Portfolio holdings
and research notes are browser-local; user databases are not exposed.

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

Browser-local favorites, drawings and settings keep their existing storage keys.
The prepared snapshot retains its existing on-disk location for cache reuse.
Native portfolio, pairs, options, liquidity and forecasting calculations also
work on daily snapshots. Fundamentals show unavailable data explicitly.

## Verification

Install `requirements.txt` and `playwright`. Browser checks use Microsoft Edge.

```powershell
python -m unittest test_quantstack_gateway test_snapshot test_research test_local_scheduler
python smoke_research_workspace.py
python smoke_quantstack.py --url http://127.0.0.1:8501
```
