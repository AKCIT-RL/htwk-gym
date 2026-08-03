"""Unit tests for mimickit/util/terrain_util.py (pure numpy, CPU)."""
import sys
from pathlib import Path

import numpy as np
import pytest

MIMICKIT_ROOT = Path(__file__).resolve().parents[2] / "MimicKit"
sys.path.insert(0, str(MIMICKIT_ROOT / "mimickit"))

from util import terrain_util


def test_heightfield_shape_dtype_bounds():
    rng = np.random.RandomState(0)
    h = terrain_util.build_uneven_heightfield(10.0, 4.0, 0.5, 0.02, rng=rng)
    assert h.dtype == np.float32
    # ceil(size / scale) + 1 vertices per axis
    assert h.shape == (21, 9)
    assert np.all(np.abs(h) <= 0.02)
    assert h.std() > 0.0  # actually random


def test_heightfield_deterministic_with_seed():
    h1 = terrain_util.build_uneven_heightfield(5.0, 5.0, 0.5, 0.05,
                                               rng=np.random.RandomState(7))
    h2 = terrain_util.build_uneven_heightfield(5.0, 5.0, 0.5, 0.05,
                                               rng=np.random.RandomState(7))
    assert np.array_equal(h1, h2)


def test_tile_border_flattened_interior_random():
    rng = np.random.RandomState(3)
    h = terrain_util.build_uneven_tile(18.0, 13.0, 0.5, 0.02, rng=rng)
    assert h.shape == (37, 27)
    # border ring at exactly z = 0 so edge-to-edge tiles meet seamlessly
    assert np.all(h[0, :] == 0.0) and np.all(h[-1, :] == 0.0)
    assert np.all(h[:, 0] == 0.0) and np.all(h[:, -1] == 0.0)
    interior = h[1:-1, 1:-1]
    assert interior.std() > 0.0
    assert np.all(np.abs(interior) <= 0.02)


def test_tile_extent_matches_pitch():
    # a tile sized to the field pitch must span exactly pitch meters so that
    # tiles laid at compute_field_offsets centers touch without gaps/overlap
    h = terrain_util.build_uneven_tile(18.0, 13.0, 0.5, 0.02,
                                       rng=np.random.RandomState(0))
    verts, _ = terrain_util.heightfield_to_trimesh(h, 0.5, x_offset=-9.0, y_offset=-6.5)
    assert np.isclose(verts[:, 0].min(), -9.0) and np.isclose(verts[:, 0].max(), 9.0)
    assert np.isclose(verts[:, 1].min(), -6.5) and np.isclose(verts[:, 1].max(), 6.5)


def test_heightfield_rejects_bad_args():
    with pytest.raises(ValueError):
        terrain_util.build_uneven_heightfield(0.0, 5.0, 0.5, 0.02)
    with pytest.raises(ValueError):
        terrain_util.build_uneven_heightfield(5.0, 5.0, 0.0, 0.02)
    with pytest.raises(ValueError):
        terrain_util.build_uneven_heightfield(5.0, 5.0, 0.5, -0.01)


def test_trimesh_counts_dtypes_and_geometry():
    heights = np.array([[0.00, 0.01],
                        [0.02, -0.01],
                        [0.03, 0.00]], dtype=np.float32)
    verts, tris = terrain_util.heightfield_to_trimesh(heights, 0.5,
                                                      x_offset=-1.0, y_offset=2.0)
    nx, ny = heights.shape
    assert verts.shape == (nx * ny, 3)
    assert tris.shape == (2 * (nx - 1) * (ny - 1), 3)
    assert verts.dtype == np.float32
    assert tris.dtype == np.uint32
    assert np.all(tris < verts.shape[0])
    # vertex (i, j) at (x_offset + i*hs, y_offset + j*hs, heights[i, j])
    assert np.allclose(verts[0], [-1.0, 2.0, 0.00])
    assert np.allclose(verts[1], [-1.0, 2.5, 0.01])
    assert np.allclose(verts[ny], [-0.5, 2.0, 0.02])
    assert np.allclose(verts[:, 2].reshape(nx, ny), heights)


def test_trimesh_winding_faces_up():
    heights = np.zeros([3, 3], dtype=np.float32)
    verts, tris = terrain_util.heightfield_to_trimesh(heights, 1.0)
    v = verts[tris]  # [T, 3, 3]
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 1])
    assert np.all(normals[:, 2] > 0.0)  # CCW seen from +z


def test_trimesh_rejects_degenerate_field():
    with pytest.raises(ValueError):
        terrain_util.heightfield_to_trimesh(np.zeros([1, 5], dtype=np.float32), 0.5)
