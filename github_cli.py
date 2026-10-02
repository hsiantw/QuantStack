"""Use GitHub CLI with credentials from the existing Git credential manager."""
import os
import shutil
import subprocess


def run(arguments, **kwargs):
    environment = os.environ.copy()
    if not environment.get('GH_TOKEN') and not environment.get('GITHUB_TOKEN'):
        credential = subprocess.run(
            ['git', '-c', 'credential.interactive=false', 'credential', 'fill'],
            input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True,
            env=dict(environment, GIT_TERMINAL_PROMPT='0'), timeout=30)
        values = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
        if credential.returncode == 0 and values.get('password'):
            environment['GH_TOKEN'] = values['password']
    executable = shutil.which('gh')
    if not executable:
        raise RuntimeError('Install GitHub CLI to publish QuantStack data.')
    return subprocess.run([executable, *arguments], env=environment, check=True, **kwargs)
