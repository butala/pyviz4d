import os
import tempfile

CACHE_PATH = os.path.join(tempfile.gettempdir(), 'pyviz4d_cache')
os.makedirs(CACHE_PATH, exist_ok=True)
