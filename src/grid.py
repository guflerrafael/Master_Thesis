"""Native SEVIRI pixel grid of the FRP extract (notes.md 4.2).

The pixel centres of the extract (irregular in lat/lon) lie on the native SEVIRI grid of the
geostationary projection with sub-satellite longitude 0 deg (all of 2004-2019):
  row = y / SSD, col = x / SSD   (rounded; row increases northwards)
Image arrays use index i = ROW_MAX - row (north at the top) and j = col - COL_MIN.
"""
import numpy as np
from pyproj import Transformer

SSD = 3000.403165817   # SEVIRI sampling distance at nadir [m]
GEOS = "+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y +units=m"
ROW_MIN, ROW_MAX = 1206, 1382     # extent of the extract (all detections)
COL_MIN, COL_MAX = -260, -155
H, W = ROW_MAX - ROW_MIN + 1, COL_MAX - COL_MIN + 1   # 177 x 106

_to_geos = Transformer.from_crs("EPSG:4326", GEOS, always_xy=True)
_from_geos = Transformer.from_crs(GEOS, "EPSG:4326", always_xy=True)


def native_rc(lat, lon):
    """Continuous native (row, col)."""
    x, y = _to_geos.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(y) / SSD, np.asarray(x) / SSD


def rc_to_latlon(row, col):
    lon, lat = _from_geos.transform(np.asarray(col, float) * SSD, np.asarray(row, float) * SSD)
    return np.asarray(lat), np.asarray(lon)


def rc_to_ij(row, col):
    """Native (row, col) -> image (i, j); works for continuous values."""
    return ROW_MAX - np.asarray(row), np.asarray(col) - COL_MIN


def ij_to_rc(i, j):
    return ROW_MAX - np.asarray(i), np.asarray(j) + COL_MIN


def latlon_to_ij(lat, lon):
    return rc_to_ij(*native_rc(lat, lon))


def ij_to_latlon(i, j):
    return rc_to_latlon(*ij_to_rc(i, j))


def cell_latlon():
    """(H, W) arrays with the lat/lon of every cell centre."""
    i, j = np.mgrid[0:H, 0:W]
    return ij_to_latlon(i, j)
