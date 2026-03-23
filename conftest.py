import sys
from pathlib import Path

# Make src importable from tests without installing the package
sys.path.insert(0, str(Path(__file__).parent))
