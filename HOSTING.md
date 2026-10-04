# QuantStack hosting and market data

The single public application is **https://quantstack.onrender.com/**.
Source and data releases belong to **https://github.com/hsiantw/QuantStack**.
The former standalone chart website is retired; no second public UI is published.

## Render setup

- Build: `pip install -r requirements.txt && python prepare_snapshot.py`
- Start: `python serve_quantstack.py --host 0.0.0.0`
- Branch: `main`, with automatic deployment enabled.

The launcher reads Render's `PORT` variable and serves the native chart workspace.
The old Streamlit start command is retired; use `python serve_quantstack.py --host 0.0.0.0`.

## Data publication

`publish_site.py` retains its filename for the existing daily collector, but now
publishes only data to QuantStack:

1. Build the daily snapshot from the local database.
2. Package the catalogs and compressed per-symbol prices in `market-data.zip`.
3. Upload to QuantStack's `market-data` release and verify GitHub's SHA-256 digest.
4. Update and push `market-snapshot.json`, triggering Render's normal deployment.
5. Render verifies and installs the archive, then serves data from its own address.

```powershell
python publish_site.py
```

`deployment.json` must contain `{"repository": "hsiantw/QuantStack"}`. Publishing
to another repository is rejected. No GitHub Pages workflow is dispatched.
GitHub CLI and existing Git authentication are required; credentials stay out of
project files. A successful push is not proof of a successful Render deployment.

Snapshots are installed into versioned directories beneath
`QuantStack-main/static/market-data/`. The checksum and archive paths are checked
before a version becomes available. Database files, local deployment settings,
and generated datasets stay outside source control. The release contains market
prices and screener metadata, not account databases or portfolio records.

Daily snapshots support charts, drawings, comparisons, screening, backtesting,
Markov analysis, and Brownian-motion simulations. Native hourly data and the full indicator library require
the shared local database. Deployment does not itself refresh prices.

`build_site.py` still creates a portable local preview used by regression tests;
that preview is not published as another website.
