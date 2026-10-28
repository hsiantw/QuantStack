"""Read-only market database inspection with bounded, parameterized filters."""
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
import sqlite3
import math


def inspect(database, query):
    path = Path(database).resolve()
    if not path.is_file():
        raise ValueError('The local market database does not exist yet.')
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        # Exclude internal SQLite tables; no user-provided SQL or paths are accepted.
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        table = query.get('table', [None])[0]
        if table is None:
            return dict(database=path.name, bytes=path.stat().st_size,
                wal_bytes=Path(str(path)+'-wal').stat().st_size if Path(str(path)+'-wal').exists() else 0,
                tables=tables, schemas={name:[dict(r) for r in db.execute('PRAGMA table_info("'+name.replace('"','""')+'")')] for name in tables}, read_only=True)
        if table not in tables:
            raise ValueError('Choose an existing market-data table.')
        quote = lambda name: '"' + name.replace('"', '""') + '"'
        schema = [dict(r) for r in db.execute(f'PRAGMA table_info({quote(table)})')]
        columns = [r['name'] for r in schema]
        where, args = [], []
        for field in ('symbol', 'interval', 'source', 'status'):
            value = query.get(field, [''])[0].strip()
            if value:
                if field not in columns:
                    raise ValueError(f'This table has no {field} column.')
                if len(value) > 100:
                    raise ValueError('Filter value is too long.')
                where.append(quote(field) + '=?')
                args.append(value.upper() if field == 'symbol' else value)
        time_column = next((c for c in ('timestamp', 'date', 'requested_at', 'started', 'observed_date') if c in columns), None)
        start, end = query.get('start', [''])[0], query.get('end', [''])[0]
        if start and end and start > end:
            raise ValueError('Start must be on or before the end date.')
        for value, lower in ((start, True), (end, False)):
            if not value:
                continue
            if time_column is None:
                raise ValueError('This table has no filterable date column.')
            instant = date.fromisoformat(value)
            if not lower:
                if instant == date.max:
                    raise ValueError('End date must be earlier than 9999-12-31.')
                instant += timedelta(days=1)
            where.append(f'{quote(time_column)} ' + ('>=?' if lower else '<?'))
            args.append(instant.isoformat())
        try:
            offset = int(query.get('offset', ['0'])[0])
            limit = int(query.get('limit', ['50'])[0])
        except ValueError:
            raise ValueError('Invalid page size or offset.') from None
        if not 0 <= offset <= 100000 or not 1 <= limit <= 200:
            raise ValueError('Use 1–200 rows per page and an offset up to 100,000; filter large tables by date.')
        clause = ' WHERE ' + ' AND '.join(where) if where else ''
        primary = [r['name'] for r in sorted(schema, key=lambda r:r['pk']) if r['pk']]
        order = ([time_column] if time_column else []) + [c for c in primary if c != time_column]
        if not order:
            order = columns[:1]
        ordering = ','.join(quote(c) + ' DESC' for c in order)
        # Deadline protects the chart server from unexpectedly large unindexed scans.
        import time
        deadline = time.monotonic() + 8
        db.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        try:
            count = db.execute(f'SELECT COUNT(*) FROM {quote(table)}{clause}', args).fetchone()[0]
            rows = [dict(r) for r in db.execute(f'SELECT * FROM {quote(table)}{clause} ORDER BY {ordering} LIMIT ? OFFSET ?', (*args, limit, offset))]
        except sqlite3.OperationalError as exc:
            if 'interrupted' in str(exc):
                raise ValueError('Query exceeded eight seconds. Narrow the symbol/date filters.') from None
            raise
        # SQLite blobs are shown as a size descriptor, never silently JSON-encoded.
        for row in rows:
            for key, value in row.items():
                if isinstance(value, bytes):
                    row[key] = f'<{len(value)} bytes>'
                elif isinstance(value, float) and not math.isfinite(value):
                    row[key] = str(value)
        return dict(database=path.name, table=table, schema=schema, columns=columns,
            filters=[c for c in ('symbol','interval','source','status') if c in columns], date_column=time_column,
            rows=rows, count=count, offset=offset, limit=limit, has_more=offset+len(rows)<count,
            read_only=True)
