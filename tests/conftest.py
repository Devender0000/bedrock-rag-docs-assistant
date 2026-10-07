import sys
from pathlib import Path

# Make the project root importable so tests can `import ingestion`, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
