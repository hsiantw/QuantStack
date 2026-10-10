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

## Account storage

The chart gateway now serves registration, login, logout and versioned personal
workspaces at `/workspace/api/account/`. Market snapshots and accounts can be
served by the same gateway. The default local account database is
`data/accounts.sqlite`; it is excluded from Git and data release archives.

Before enabling public accounts, configure:

- `QUANTSTACK_PUBLIC_ORIGIN=https://quantstack.onrender.com` (or the exact HTTPS
  origin for this instance, with no trailing slash). This sets Secure cookies
  and the allowed origin for account writes. Terminate HTTPS at a trusted reverse
  proxy and prevent direct public access to the backend HTTP port.
- `QUANTSTACK_ACCOUNTS_DB=/var/data/quantstack/accounts.sqlite`, with `/var/data`
  mounted on durable storage. Back up this database with SQLite's backup API.
  All processes serving an instance must use the same account database; this
  SQLite implementation is intended for a single server, not separate replicas.

The checked-in free Render configuration does not provision durable account
storage. Do not enable public accounts on an ephemeral filesystem: users and
saved work would disappear on restart or deployment. No hosting plan or paid
disk is provisioned by this code change. Without HTTPS/origin configuration,
HTTP account access is restricted to the local computer.

Users explicitly save/load personal workspace snapshots, up to 2 MB per account.
Password hashes and session records stay server-side; session tokens are carried
only in HttpOnly cookies. Authentication throttles apply per connection peer and
username; users behind the same reverse proxy may share the peer limit.
There is no email verification, password recovery, or automatic migration from
the retired Streamlit account database. Accounts on separate app instances are
separate. Static HTML previews cannot create or store accounts.

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

## Refreshing stale charts

`python refresh_all.py --workers 6` catches up the entire configured daily and
hourly universe. It interleaves popular stocks, US market-cap and share-volume
leaders, and configured cryptocurrencies before the rest of the universe.
It preserves existing history, excludes open candles, stops on rate limiting,
and records successes and failures in `data/all-refresh.json`. Use `--resume`
to retry unfinished symbols from the same day's run. Daily/weekly/monthly charts
are published; hourly bars remain available in the local dashboard.

The bounded scheduled collectors use `data/refresh-priority.json` to put due
priority assets ahead of the normal oldest-attempt rotation. `Install-Schedule.ps1`
now runs `Refresh-And-Publish.ps1`: the configured low-resource daily collection
budget is retained, then successful updates are packaged and published to Render.
This requires the computer to be on and the user signed in.

Publication uses a dedicated clean checkout at `../market-data-render-deploy`:

```powershell
git worktree add -b deploy/market-data-refresh ../market-data-render-deploy origin/main
python publish_site.py --deployment-worktree ../market-data-render-deploy
```

The publisher fast-forwards that checkout to the remote deployment branch and
pushes only the snapshot manifest. It rejects dirty checkouts or unrelated
unpublished commits, so local source checkpoints cannot accidentally be deployed.
