"""Locate the model data included in both editable and wheel installations."""

from pathlib import Path


def get_asset_root() -> Path:
    """Return the package directory containing ``model/``.

    This is also suitable for consumers accepting an upstream checkout root
    (for example, ``--hack-dir``).
    """
    return Path(__file__).resolve().parent


def get_model_dir() -> Path:
    """Return the directory containing S.npy, the OBJ templates, and rig data."""
    return get_asset_root() / "model"
