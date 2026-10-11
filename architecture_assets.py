"""Explicit public assets for the versioned project architecture guide."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHITECTURE_ASSETS = {
    'architecture.html': (ROOT / 'web' / 'architecture.html', 'text/html'),
    'architecture.css': (ROOT / 'web' / 'architecture.css', 'text/css'),
    'architecture.js': (ROOT / 'web' / 'architecture.js', 'text/javascript'),
    'architecture-data.js': (ROOT / 'web' / 'architecture-data.js', 'text/javascript'),
    'project-map.md': (ROOT / 'docs' / 'project-map.md', 'text/markdown'),
    'project-tree.txt': (ROOT / 'docs' / 'project-tree.txt', 'text/plain'),
    'project-git-objects.tsv': (ROOT / 'docs' / 'project-git-objects.tsv', 'text/tab-separated-values'),
}
