import logging

logger = logging.getLogger('pyviz4d.blue_marble')

# BMNG dataset: http://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74368/
URL_MAP = {
    'low':    'https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74368/world.topo.200406.3x5400x2700.jpg',
    'medium': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74368/world.topo.200406.3x21600x10800.jpg'
}

def fetch(path, resolution='low'):
    """
    Fetch the Blue Marble texture from NASA servers using Pooch.
    Pooch will automatically cache the file and skip downloading if it already exists.
    """
    # Imported lazily so that ``import pyviz4d`` does not require pooch: only
    # the Earth texture path needs it, and non-Earth consumers (e.g. the
    # SphericalCT viz layer) do not have it installed.
    import pooch

    try:
        url = URL_MAP[resolution]
    except KeyError:
        raise ValueError('resolution must be one of: {}'.format(' '.join(sorted(URL_MAP))))

    logger.info(f"Ensuring {resolution} Blue Marble texture is available in {path}")

    # pooch.retrieve automatically handles checking the cache and downloading if necessary.
    # We set known_hash=None to just trust the URL without requiring a hardcoded SHA,
    # and fname extracts the filename from the URL automatically.
    local_fname = pooch.retrieve(
        url=url,
        known_hash=None,
        path=path,
        fname=url.split('/')[-1]
    )

    return local_fname
