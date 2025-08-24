import sys
from pathlib import Path


def pytest_sessionstart(session):
    """Ensure project root is on sys.path for absolute imports like 'src.ner.*'."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


