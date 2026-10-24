"""VTU loading and property / field argument helpers."""

from __future__ import annotations

from typing import Any

import meshio
import numpy as np


def read_vtu(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read VTU points (n, 3), tetrahedra (m, 4), and PointData ``value`` (scalar or 3D vector)."""
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=float)[:, :3]
    if "tetra" not in mesh.cells_dict:
        raise ValueError(f"VTU {path!r} has no tetrahedron cells")
    tet = np.asarray(mesh.cells_dict["tetra"], dtype=int)
    if "value" not in mesh.point_data:
        raise ValueError(f"VTU {path!r} has no PointData array named 'value'")
    values = np.asarray(mesh.point_data["value"], dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim == 2 and values.shape[1] >= 3:
        values = values[:, :3]
    elif values.ndim != 1:
        raise ValueError(f"Unsupported PointData 'value' shape {values.shape} in {path!r}")
    return points, tet, values


def eval_prop_scl(prop: Any, unk: np.ndarray | None, n: int, name: str) -> np.ndarray:
    """Vertex values of a scalar property: constant, (n,) array, or ``Callable[[float], float]`` of the unknown."""
    if callable(prop) and not np.isscalar(prop):
        if unk is None:
            raise TypeError(f"{name} cannot be a callable here; pass a float or a per-vertex array")
        return np.array([float(prop(u)) for u in unk], dtype=float)
    arr = np.asarray(prop, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a float or have shape ({n},), got {arr.shape}")
    return arr


def eval_prop_vec(prop: Any, n: int, name: str) -> np.ndarray:
    """Vertex values of a vector property: (3,) constant or (n, 3) array."""
    arr = np.asarray(prop, dtype=float)
    if arr.shape == (3,):
        return np.tile(arr, (n, 1))
    if arr.shape != (n, 3):
        raise ValueError(f"{name} must have shape (3,) or ({n}, 3), got {arr.shape}")
    return arr
