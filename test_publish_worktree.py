import subprocess
import unittest
from unittest.mock import patch
from pathlib import Path

from publish_site import publish_manifest


class PublishWorktreeTests(unittest.TestCase):
    def test_rejects_dirty_checkout_without_fetching_or_pushing(self):
        def run(args, **kwargs):
            output = 'https://github.com/hsiantw/QuantStack.git\n' if 'get-url' in args else ' M web/app.js\n'
            return subprocess.CompletedProcess(args, 0, stdout=output)
        with patch('publish_site.subprocess.run', side_effect=run) as command:
            with self.assertRaisesRegex(RuntimeError, 'local changes'):
                publish_manifest({}, Path.cwd())
            self.assertEqual(command.call_count, 2)

    def test_rejects_unpublished_source_commits(self):
        def run(args, **kwargs):
            output = 'https://github.com/hsiantw/QuantStack.git\n' if 'get-url' in args else ''
            if '--name-only' in args:
                output = 'web/app.js\n'
            return subprocess.CompletedProcess(args, 0, stdout=output)
        with patch('publish_site.subprocess.run', side_effect=run) as command:
            with self.assertRaisesRegex(RuntimeError, 'unpublished source'):
                publish_manifest({}, Path.cwd())
            self.assertFalse(any('push' in c.args[0] for c in command.call_args_list))


if __name__ == '__main__':
    unittest.main()
