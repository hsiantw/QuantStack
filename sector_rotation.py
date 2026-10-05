"""Cached sector performance snapshots from the Finviz Group Screener."""
import math
import threading
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCE_URL = 'https://finviz.com/groups?g=sector&v=140'
CACHE_SECONDS = 300
_cache = None
_cache_time = 0.0
_cache_lock = threading.Lock()


class SectorRotationError(RuntimeError):
    """Finviz data could not be fetched or parsed."""


class _GroupsTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_groups_table = False
        self.table_depth = 0
        self.rows = []
        self.row = None
        self.cell = None
        self.cell_tag = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'table':
            if not self.in_groups_table and 'groups_table' in attributes.get('class', '').split():
                self.in_groups_table = True
                self.table_depth = 1
            elif self.in_groups_table:
                self.table_depth += 1
        elif self.in_groups_table and self.table_depth == 1:
            if tag == 'tr':
                if self.row:
                    self.rows.append(self.row)
                self.row = []
            elif self.row is not None and self.cell is None and tag in ('th', 'td'):
                self.cell, self.cell_tag = [], tag

    def handle_endtag(self, tag):
        if tag == 'table' and self.in_groups_table:
            self.table_depth -= 1
            if self.table_depth == 0:
                self.in_groups_table = False
        elif self.in_groups_table and self.table_depth == 1:
            if self.cell is not None and tag == self.cell_tag:
                self.row.append(' '.join(''.join(self.cell).split()))
                self.cell, self.cell_tag = None, None
            elif self.row is not None and tag == 'tr':
                if self.row:
                    self.rows.append(self.row)
                self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def _parse_percent(value):
    try:
        number = float(value.strip().replace('%', '').replace(',', ''))
    except (AttributeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def parse_groups(html):
    parser = _GroupsTableParser()
    parser.feed(html)
    if len(parser.rows) < 2:
        raise SectorRotationError('Finviz did not return a sector performance table.')
    columns = {name.casefold(): index for index, name in enumerate(parser.rows[0])}
    fields = {
        'week_pct': 'perf week',
        'month_pct': 'perf month',
        'quarter_pct': 'perf quart',
        'half_year_pct': 'perf half',
        'year_pct': 'perf year',
        'ytd_pct': 'perf ytd',
        'change_1d_pct': 'change %',
    }
    if not all(name in columns for name in ('name', *fields.values())):
        raise SectorRotationError('Finviz sector performance table has an unsupported format.')
    sectors = []
    for values in parser.rows[1:]:
        if len(values) < len(parser.rows[0]):
            continue
        name = values[columns['name']]
        if not name:
            continue
        sector = {'name': name}
        for key, column in fields.items():
            sector[key] = _parse_percent(values[columns[column]])
        sectors.append(sector)
    if not sectors:
        raise SectorRotationError('Finviz returned no sector performance rows.')
    for key in ('week_pct', 'quarter_pct'):
        ranked = sorted(
            (item for item in sectors if item[key] is not None),
            key=lambda item: item[key], reverse=True,
        )
        previous_value = None
        rank = 0
        for index, item in enumerate(ranked, 1):
            if index == 1 or item[key] != previous_value:
                rank = index
            item[key.replace('_pct', '_rank')] = rank
            previous_value = item[key]
    for item in sectors:
        week_rank, quarter_rank = item.get('week_rank'), item.get('quarter_rank')
        if week_rank is None or quarter_rank is None:
            item['rotation'] = 'Unavailable'
        elif quarter_rank - week_rank >= 2:
            item['rotation'] = 'Improving'
        elif quarter_rank - week_rank <= -2:
            item['rotation'] = 'Cooling'
        else:
            item['rotation'] = 'Steady'
    return sectors


def _fetch_groups():
    request = Request(SOURCE_URL, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; QuantStack/1.0)',
        'Accept': 'text/html',
    })
    try:
        with urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise SectorRotationError(f'Finviz returned HTTP {response.status}.')
            html = response.read().decode('utf-8', errors='replace')
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SectorRotationError(f'Could not retrieve sector data from Finviz: {exc}') from exc
    return parse_groups(html)


def snapshot():
    """Return a cached, source-attributed sector performance snapshot."""
    global _cache, _cache_time
    with _cache_lock:
        now = datetime.now(timezone.utc).timestamp()
        if _cache is not None and now - _cache_time < CACHE_SECONDS:
            return dict(_cache)
        sectors = _fetch_groups()
        _cache = {
            'source': SOURCE_URL,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sectors': sectors,
        }
        _cache_time = now
        return dict(_cache)
