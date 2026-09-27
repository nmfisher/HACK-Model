"""HACK assets and optional PyTorch model, independent of the calling project."""

from .assets import get_asset_root, get_model_dir

__all__ = ["HACK", "PCA", "load_pca", "get_asset_root", "get_model_dir",
           "HACKIdentityBasis", "load_identity_basis"]


def __getattr__(name):
    if name in {"HACKIdentityBasis", "load_identity_basis"}:
        from . import identity

        value = getattr(identity, name)
        globals()[name] = value
        return value
    # Asset-only consumers (including Blender) do not need torch or OpenCV.
    if name in {"HACK", "PCA", "load_pca"}:
        from . import core

        value = getattr(core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
