import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import dashboard
import sector_rotation


class SectorRotationTests(unittest.TestCase):
    @staticmethod
    def groups_html():
        headers = ['No.', 'Name', 'Perf Week', 'Perf Month', 'Perf Quart',
                   'Perf Half', 'Perf Year', 'Perf YTD', 'Avg Volume',
                   'Rel Volume', 'Change %', 'Volume']
        rows = [
            ['1', 'Alpha', '8.00%', '4.00%', '4.00%', '5.00%', '6.00%', '3.00%', '1M', '1.0', '0.2%', '1M'],
            ['2', 'Beta', '6.00%', '3.00%', '2.00%', '4.00%', '5.00%', '2.00%', '1M', '1.0', '0.1%', '1M'],
            ['3', 'Gamma', '4.00%', '2.00%', '12.00%', '3.00%', '4.00%', '1.00%', '1M', '1.0', '0.1%', '1M'],
            ['4', 'Delta', '0.00%', '1.00%', '8.00%', '2.00%', '3.00%', '0.00%', '1M', '1.0', '0.0%', '1M'],
            ['5', 'Epsilon', '-2.00%', '0.00%', '1.00%', '1.00%', '2.00%', 'N/A', '1M', '1.0', '-0.1%', '1M'],
            ['6', 'Feta', '8.00%', '0.00%', '-1.00%', '0.00%', '1.00%', '0.00%', '1M', '1.0', '0.0%', '1M'],
        ]
        head = ''.join(f'<th>{value}</th>' for value in headers)
        body = ''.join('<tr>' + ''.join(f'<td><span>{value}</span></td>' for value in row) + '</tr>'
                       for row in rows)
        return f'<table class="styled-table-new groups_table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    def test_parses_sector_horizons_and_weekly_rank_rotation(self):
        sectors = sector_rotation.parse_groups(self.groups_html())
        by_name = {sector['name']: sector for sector in sectors}
        self.assertEqual(len(sectors), 6)
        self.assertEqual((by_name['Alpha']['week_rank'], by_name['Alpha']['quarter_rank'],
                          by_name['Alpha']['rotation']), (1, 3, 'Improving'))
        self.assertEqual(by_name['Feta']['week_rank'], 1)
        self.assertEqual(by_name['Beta']['rotation'], 'Steady')
        self.assertEqual(by_name['Delta']['rotation'], 'Cooling')
        self.assertEqual(by_name['Epsilon']['rotation'], 'Steady')
        self.assertIsNone(by_name['Epsilon']['ytd_pct'])
        self.assertEqual(by_name['Alpha']['change_1d_pct'], 0.2)

    def test_rejects_unrecognized_group_table(self):
        with self.assertRaises(sector_rotation.SectorRotationError):
            sector_rotation.parse_groups('<html><table class="groups_table"><tr><td>Unexpected</td></tr></table></html>')

    def test_snapshot_caches_provider_response(self):
        with patch.object(sector_rotation, '_cache', None), \
             patch.object(sector_rotation, '_cache_time', 0), \
             patch.object(sector_rotation, '_fetch_groups', return_value=[{'name': 'Technology'}]) as fetch:
            first = sector_rotation.snapshot()
            second = sector_rotation.snapshot()
        fetch.assert_called_once()
        self.assertEqual(first, second)
        self.assertEqual(first['source'], sector_rotation.SOURCE_URL)

    def test_api_returns_finviz_snapshot_and_provider_errors(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}/api/sector-rotation'
        response = {'source': sector_rotation.SOURCE_URL, 'generated_at': '2026-01-01T00:00:00+00:00', 'sectors': []}
        with patch.object(sector_rotation, 'snapshot', return_value=response):
            with urllib.request.urlopen(base) as result:
                self.assertEqual(json.loads(result.read()), response)
        with patch.object(sector_rotation, 'snapshot',
                          side_effect=sector_rotation.SectorRotationError('Finviz is unavailable.')):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base)
            self.assertEqual(error.exception.code, 502)
            self.assertEqual(json.loads(error.exception.read())['error'], 'Finviz is unavailable.')


if __name__ == '__main__':
    unittest.main()
