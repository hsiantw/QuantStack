import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from prepare_snapshot import prepare


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / 'manifest.json'
        self.archive = self.root / 'market-data.zip'
        self.output = self.root / 'static'

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, extra=None, missing=None):
        files = {'symbols.json': '[]', 'screener.json': '{}', 'snapshot.json': '{}',
                 'prices/AAPL.json.gz': 'price fixture', 'prices/%5EGSPC.json.gz': 'index fixture'}
        files.update(extra or {})
        files.pop(missing, None)
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name, value in files.items():
                archive.writestr(name, value)
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps({'sha256': digest, 'url': 'https://example.invalid/data.zip'}))
        return digest

    def test_verified_atomic_install_and_offline_reuse(self):
        digest = self.fixture()
        installed = prepare(self.manifest, self.output, self.archive)
        self.assertEqual(installed.name, digest[:16])
        self.assertEqual((installed / 'prices/^GSPC.json.gz').read_text(), 'index fixture')
        self.assertEqual((installed / '.complete').read_text(), digest)
        with patch('urllib.request.urlopen', side_effect=AssertionError('Must reuse installed data')):
            self.assertEqual(prepare(self.manifest, self.output), installed)

    def test_checksum_failure_does_not_publish_partial_data(self):
        self.fixture()
        self.archive.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            prepare(self.manifest, self.output, self.archive)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_rejects_unexpected_paths_and_encoded_traversal(self):
        for name in ('../users.db', 'prices/../../users.db', 'prices/%2E%2E%2Fsecret.json.gz', 'index.html'):
            with self.subTest(name=name):
                self.fixture({name: 'unwanted'})
                with self.assertRaisesRegex(ValueError, 'Unexpected files'):
                    prepare(self.manifest, self.output, self.archive)
                self.assertEqual(list(self.output.iterdir()), [])

    def test_requires_all_catalogs(self):
        self.fixture(missing='screener.json')
        with self.assertRaisesRegex(ValueError, 'missing its catalogs'):
            prepare(self.manifest, self.output, self.archive)


if __name__ == '__main__':
    unittest.main()
