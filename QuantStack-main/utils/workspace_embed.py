"""Bundle the merged workspace for hosts using Streamlit's original start command."""
import json
from pathlib import Path
import re

WEB = Path(__file__).resolve().parents[2] / 'web'


def snapshot_html(snapshot_url):
    def script(source):
        return '<script>' + source.replace('</script', '<\\/script') + '</script>'

    html = (WEB / 'index.html').read_text(encoding='utf-8-sig')
    worker = (WEB / 'markov-engine.js').read_text(encoding='utf-8-sig') + '\n' + (
        WEB / 'markov-worker.js').read_text(encoding='utf-8-sig').replace("importScripts('./markov-engine.js');", '')
    setup = 'window.ATLAS_DATA_BASE=' + json.dumps(snapshot_url.rstrip('/') + '/') + ';\n'
    setup += 'window.ATLAS_MARKOV_WORKER_URL=URL.createObjectURL(new Blob([' + json.dumps(worker) + '],{type:"text/javascript"}));'
    adapter = script(setup) + script((WEB / 'static-data.js').read_text(encoding='utf-8-sig'))

    def inline_script(match):
        name = match.group(1)
        return (adapter if name == 'app.js' else '') + script((WEB / name).read_text(encoding='utf-8-sig'))

    html = re.sub(r'<script src="\./([a-z-]+\.js)"></script>', inline_script, html)
    html = re.sub(r'<link rel="stylesheet" href="\./([a-z-]+\.css)">',
                  lambda match: '<style>' + (WEB / match.group(1)).read_text(encoding='utf-8-sig') + '</style>', html)
    return html
