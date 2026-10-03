"""read_cityjson() against a known-good file.

The fixture is tests/data/sample_building.city.json: a 10 m cube in spec
nesting, small enough to reason about by hand.
"""
import json
from pathlib import Path

import pytest

from pyviz4d import read_cityjson

DATA = Path(__file__).parent / "data"


def test_read_cityjson_unit_box():
    """A Solid is boundaries -> shell -> surface -> ring.

    One six-faced cube therefore comes out as 8 points and 6 polygons.  A file
    written one nesting level short (boundaries = [surface, ...], which is what
    the demo_lod1_* generators used to emit) raises
    ``TypeError: object of type 'int' has no len()`` here instead.
    """
    pd = read_cityjson(str(DATA / "sample_building.city.json"))
    assert pd.GetNumberOfPoints() == 8
    assert pd.GetNumberOfPolys() == 6


def test_read_cityjson_rejects_short_nesting(tmp_path):
    """The same cube with the shell level collapsed must not read.

    This is the regression the demo_lod1_* generators used to trip: they wrote
    ``boundaries = [surface, ...]`` where the spec wants ``[shell]`` and the
    ring indices then sat where a surface list belongs.
    """
    good = json.loads((DATA / "sample_building.city.json").read_text())
    solid = good["CityObjects"]["Building1"]["geometry"][0]
    solid["boundaries"] = solid["boundaries"][0]        # drop the shell level
    path = tmp_path / "short.city.json"
    path.write_text(json.dumps(good))

    with pytest.raises(TypeError):
        read_cityjson(str(path))
