"""WGS84 -> Cartesian maths -- the numbers ``examples/validate_wgs84.py`` draws.

That demo puts two markers at the North Pole to show the polar flattening
visually; these are the same values checked numerically.
"""
import pytest

from pyviz4d.earth import WGS84, Ellipsoid, sphere_to_cartesian, wgs84_to_cartesian


def test_wgs84_ellipsoid_shape():
    assert WGS84.a == pytest.approx(6378137.0)                 # equatorial, m
    assert WGS84.f == pytest.approx(1 / 298.257223563)
    assert WGS84.b == pytest.approx(6356752.3142, abs=1e-3)    # polar, m


def test_pole_lands_on_the_polar_radius():
    x, y, z = wgs84_to_cartesian(90.0, 0.0)                   # returns km
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert z == pytest.approx(WGS84.b / 1e3, rel=1e-12)


def test_equator_prime_meridian_lands_on_a():
    x, y, z = wgs84_to_cartesian(0.0, 0.0)
    assert (x, y, z) == pytest.approx((WGS84.a / 1e3, 0.0, 0.0))


def test_prime_meridian_is_the_xz_plane():
    x, y, z = wgs84_to_cartesian(45.0, 0.0)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert x > 0.0 and z > 0.0


def test_altitude_adds_to_the_radius():
    x, _, _ = wgs84_to_cartesian(0.0, 0.0, alt_km=100.0)
    assert x == pytest.approx(WGS84.a / 1e3 + 100.0)


def test_sphere_overshoots_the_pole_by_the_flattening():
    """What validate_wgs84.py is about: a sphere of radius `a` is too tall."""
    z_wgs = wgs84_to_cartesian(90.0, 0.0)[2]
    z_sph = sphere_to_cartesian(90.0, 0.0, radius_km=WGS84.a / 1e3)[2]
    assert z_sph == pytest.approx(WGS84.a / 1e3)
    assert z_sph - z_wgs == pytest.approx(21.38, abs=0.01)     # km


def test_sphere_matches_the_ellipsoid_at_the_equator():
    for lon in (0.0, 30.0, -120.0):
        assert sphere_to_cartesian(0.0, lon, radius_km=WGS84.a / 1e3) == \
            pytest.approx(wgs84_to_cartesian(0.0, lon))


def test_ellipsoid_derives_f_and_b():
    e = Ellipsoid(a=10.0, f_inv=2.0)               # f = 0.5 -> b = 5
    assert e.f == pytest.approx(0.5)
    assert e.b == pytest.approx(5.0)
