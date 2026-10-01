# Hourly collection capacity ? 2026-10-01

Routine minute collection has been replaced with native hourly collection. All **4,841 symbols** completed successfully; **1,952** were newly added. The Windows `MarketData-IntradaySync` task now runs hourly. Archived minute data was preserved and was not updated during this run.

| Measurement | Result |
| --- | ---: |
| Start (Asia/Taipei) | 2026-10-01 19:37:05 |
| Finish (Asia/Taipei) | 2026-10-01 19:47:49 |
| Total wall time | 644.34 seconds / 10.74 minutes |
| Existing symbols refreshed | 2,889 |
| New symbols backfilled | 1,952 |
| Stored hourly symbols after run | 4,841 |
| Stored hourly candles after run | 7,562,638 |
| Validated candles upserted this run, including overlap | 2,806,545 |
| History-call attempts | 4,841 |
| Failed calls / rate-limit responses | 0 / 0 |
| Successful symbols per minute | 450.79 |
| Mean history-call duration | 180.1 ms |
| 95th percentile history-call duration | 375 ms |

## Capacity number

At this measured pace, **27,048 symbols is the first projected count exceeding one hour**. This is a linear extrapolation, not an observed provider rate-limit boundary.

- Raw one-hour capacity: floor(4,841 / 644.34 ? 3,600) = **27,047 symbols**.
- Planning capacity: floor(4,841 / 644.34 ? 3,300 ? 0.8) = **19,834 symbols**. This leaves five minutes outside the collection budget and 20% headroom within it.
- Current coverage of 4,841 symbols uses about 24.4% of that planning capacity.

Use **about 19,800 symbols per hourly cycle** for planning under comparable conditions. The benchmark includes 1,952 initial history downloads plus 2,889 incremental refreshes, six workers, a 0.5-second pause per attempt per worker, validation, progress writes and SQLite commits. A pure incremental run may be faster. This test actually exercised 4,841 symbols, not 27,048, and did not establish a sustained daily quota or the provider's hard throttle point.

Yahoo availability, changing latency, symbol errors and rate limiting can reduce capacity. The [yfinance HTTP implementation](https://github.com/ranaroussi/yfinance/blob/main/yfinance/_http.py) explicitly notes that Yahoo may rate-limit or block the client. History calls may perform additional internal HTTP requests; attempt counts are not raw network-request counts. The collector stops new work on rate-limit errors and preserves completed data. The daily job shares the ingestion lock and can delay scheduled hourly runs; the throughput estimate assumes that lock is available.

## Verification

All 4,841 hourly states have a successful timestamp from this run, and SQLite contains hourly candles for every one of them. The minute table's latest fetch predates this run. Twenty-nine relevant unit tests passed; the final quote change also passed all eight dashboard tests. Browser checks passed for native hourly stock/crypto charts, hourly-only symbol selection, 1D/1H controls, screener navigation and static-mode behavior.

Charts with only hourly history open on 1H; daily history is not fabricated. The existing hosted snapshot remains daily-only. No site was published.

The completed measurement is preserved in `data/hourly-runs.jsonl`; live progress and future estimates are in `data/hourly-refresh.json`. The next scheduled update after installation is 2026-10-01 20:37:34 Asia/Taipei, then hourly while the computer is on and the user is logged in.
