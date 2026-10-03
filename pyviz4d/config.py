"""Cache location for downloaded assets (Blue Marble, ...).

This module deliberately does not touch the filesystem.  Creating a directory
because someone imported the package is a rude surprise in a sandbox or on a
read-only volume, and it is not needed: pooch creates the cache directory when
it retrieves into it.
"""
import os
import tempfile

CACHE_PATH = os.path.join(tempfile.gettempdir(), 'pyviz4d_cache')
