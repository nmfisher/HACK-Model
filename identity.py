"""NumPy-only identity assets, shared by offline and Blender consumers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import get_asset_root
from .helper import read_obj


@dataclass(frozen=True)
class HACKIdentityBasis:
    """Native-frame mean (V,3), PCA modes (K,V,3), and triangles (F,3)."""

    mean: np.ndarray
    vt_std: np.ndarray
    faces: np.ndarray


def load_identity_basis(asset_root=None) -> HACKIdentityBasis:
    """Load trusted HACK assets without importing the Torch rig model.

    ``asset_root`` contains ``model/``; omitted, use the installed assets.
    Preserve native vertex IDs, PCA scale, and triangle winding/order. Quads
    use the existing 0-2 diagonal. The upstream S.npy is a pickled dictionary:
    only use trusted model assets.
    """
    root = (get_asset_root() if asset_root is None
            else Path(asset_root).expanduser().resolve())
    pca = np.load(root / "model" / "S.npy", allow_pickle=True).item()
    mean = np.asarray(pca["mean"], dtype=np.float64)
    modes = np.asarray(pca["VT_std"], dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != 3 or not len(mean):
        raise ValueError("HACK mean must have shape (V, 3) with V > 0")
    if modes.ndim != 3 or modes.shape[1:] != mean.shape or not len(modes):
        raise ValueError("HACK VT_std must have shape (K, V, 3) with K > 0")
    if not np.isfinite(mean).all() or not np.isfinite(modes).all():
        raise ValueError("HACK identity arrays must be finite")
    mesh = read_obj(root / "model" / "000_generic_neutral_mesh.obj", tri=True)
    faces = np.asarray(mesh.fvs, dtype=np.int32)
    if len(mesh.vs) != len(mean):
        raise ValueError("HACK topology and PCA vertex counts differ")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("HACK topology must contain triangles")
    if faces.min() < 0 or faces.max() >= len(mean):
        raise ValueError("HACK topology vertex index is out of range")
    return HACKIdentityBasis(mean=mean, vt_std=modes, faces=faces)
