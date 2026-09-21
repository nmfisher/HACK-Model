"""Contracts for consumers of the installed package."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from hack_model import get_asset_root, get_model_dir, helper


class AssetTests(unittest.TestCase):
    def test_assets_available_without_model_dependencies_or_parent_project(self):
        code = """
import importlib.abc
import sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'shared', 'torch', 'cv2', 'roma'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockImports())
from hack_model import get_model_dir, helper
import numpy as np
model_dir = get_model_dir()
pca = np.load(model_dir / 'S.npy', allow_pickle=True).item()
assert pca['mean'].shape == (14062, 3)
assert pca['VT_std'].shape == (200, 14062, 3)
mesh = helper.read_obj(model_dir / '000_generic_neutral_mesh_newuv.obj')
assert mesh.vs.shape == (14062, 3)
assert mesh.fvs.shape == (14034, 4)
assert mesh.fvts.shape == mesh.fvs.shape
"""
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable, "-I", "-c", code], cwd=cwd, check=True)

    def test_asset_root_matches_legacy_directory_contract(self):
        self.assertEqual(get_asset_root() / "model", get_model_dir())
        for name in (
            "S.npy", "E.npy", "P.npy", "blendshape.npy", "bones_neutral.json",
            "weight_map_smooth.npy", "Lc_mid.png", "ts_larynx.npy",
            "000_generic_neutral_mesh.obj", "000_generic_neutral_mesh_newuv.obj",
        ):
            with self.subTest(name=name):
                self.assertTrue((get_model_dir() / name).is_file())


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "cv2", "roma")),
    "install hack-model[model] for PyTorch tests",
)
class ModelTests(unittest.TestCase):
    def test_forward_from_unrelated_working_directory_and_gradients(self):
        import torch
        from hack_model import HACK, load_pca

        torch.set_num_threads(2)
        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as cwd:
                os.chdir(cwd)
                model = HACK()
                shape = load_pca(get_model_dir() / "S.npy")
                self.assertEqual(tuple(shape().shape), (1, 14062, 3))
                theta = torch.zeros(1, 8, 3, requires_grad=True)
                tau = torch.zeros(1, 1, requires_grad=True)
                alpha = torch.zeros(1, 1)
                bsw = torch.zeros(1, 55, requires_grad=True)
                result = model(theta, tau, alpha, bsw)["T_transformed"]
                np.testing.assert_allclose(
                    result.detach().numpy(), model.T[None].numpy(), atol=1e-4)
                result.square().mean().backward()
                for value in (theta, tau, bsw):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                # The UV sampling cache must work after changing model dtype.
                model.double()
                moved = model(theta.double(), tau.double(), alpha.double(),
                              bsw.double())["T_transformed"]
                self.assertEqual(moved.dtype, torch.float64)
                self.assertTrue(torch.isfinite(moved).all())
        finally:
            os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
