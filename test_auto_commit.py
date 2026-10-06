"""Exercise scheduled pushes with isolated local Git repositories; no network."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AutoCommitTests(unittest.TestCase):
    def test_push_retry_and_preserve_manual_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            remote = root / 'remote.git'
            repo.mkdir()
            def git(*args, cwd=repo):
                return subprocess.check_output(['git', *args], cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
            git('init', '--bare', str(remote))
            git('init', '-b', 'main')
            git('config', 'user.name', 'Automation test')
            git('config', 'user.email', 'automation@example.invalid')
            git('config', 'core.autocrlf', 'false')
            git('remote', 'add', 'origin', str(remote))
            shutil.copy(Path(__file__).with_name('Auto-Commit.ps1'), repo)
            interpreter = sys.executable.replace("'", "''")
            (repo / 'Resolve-CollectorPython.ps1').write_text(f"$collectorPython = '{interpreter}'\n")
            (repo / 'snapshot_charts.py').write_text('pass\n')
            (repo / '.gitignore').write_text('data/\n')
            (repo / 'chart-snapshots').mkdir()
            (repo / 'chart-snapshots' / 'daily.csv').write_text('symbol,close\nTEST,10\n')
            (repo / 'chart-snapshots' / 'manifest.json').write_text('{}')
            source = repo / 'source.txt'
            source.write_text('before\n')
            git('add', '.')
            git('commit', '-m', 'fixture')
            def run(push=True):
                result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive',
                    '-ExecutionPolicy', 'Bypass', '-File', str(repo / 'Auto-Commit.ps1'),
                    *(['-Push'] if push else [])], cwd=repo, capture_output=True, text=True)
                return result
            source.write_text('after\n')
            result = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git('rev-parse', 'HEAD'), git('rev-parse', 'main', cwd=remote))
            self.assertFalse(git('status', '--porcelain'))
            # Commit with no push, then verify a no-change run still pushes it.
            source.write_text('retry\n')
            result = run(push=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotEqual(git('rev-parse', 'HEAD'), git('rev-parse', 'main', cwd=remote))
            result = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(git('rev-parse', 'HEAD'), git('rev-parse', 'main', cwd=remote))
            source.write_text('manual\n')
            git('add', 'source.txt')
            staged = git('diff', '--cached')
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(git('diff', '--cached'), staged)


if __name__ == '__main__':
    unittest.main()
