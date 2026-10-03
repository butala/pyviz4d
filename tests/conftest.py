"""Put ``examples/`` on ``sys.path``.

The demo_lod1_* generators share ``examples/_lod1.py`` and are run as scripts,
so Python already puts ``examples/`` on the path for them.  Their tests need
the same access.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
