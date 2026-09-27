"""The identity-only API must preserve arrays without loading the rig model."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hack_model import get_asset_root, load_identity_basis


class IdentityTests(unittest.TestCase):
    def test_real_assets_unchanged(self):
        root = get_asset_root()
        asset = load_identity_basis()
        pca = np.load(root / "model/S.npy", allow_pickle=True).item()
        np.testing.assert_array_equal(asset.mean, pca["mean"])
        np.testing.assert_array_equal(asset.vt_std, pca["VT_std"])
        # Independent legacy triangulation; exact order is part of the API.
        faces = []
        for line in (root / "model/000_generic_neutral_mesh.obj").read_text().splitlines():
            if line.startswith("f "):
                ids = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                self.assertIn(len(ids), (3, 4))
                faces.append(ids[:3])
                if len(ids) == 4:
                    faces.append([ids[0], ids[2], ids[3]])
        np.testing.assert_array_equal(asset.faces, faces)
        self.assertEqual(asset.mean.dtype, np.float64)
        self.assertEqual(asset.vt_std.dtype, np.float64)
        self.assertEqual(asset.faces.dtype, np.int32)

    def test_explicit_root_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model/000_generic_neutral_mesh.obj").write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                "f 1 2 3 4\nf 1 3 4\n")
            mean = np.zeros((4, 3))
            modes = np.ones((2, 4, 3))
            np.save(root / "model/S.npy", {"mean": mean, "VT_std": modes})
            asset = load_identity_basis(root)
            np.testing.assert_array_equal(asset.faces, [[0,1,2],[0,2,3],[0,2,3]])
            for invalid in (np.zeros((2, 3, 3)), np.full((2, 4, 3), np.nan)):
                np.save(root / "model/S.npy", {"mean": mean, "VT_std": invalid})
                with self.assertRaises(ValueError):
                    load_identity_basis(root)

    def test_no_heavy_or_application_dependencies(self):
        code = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'cv2', 'roma', 'bpy', 'faces',
                                     'shared', 'prediction_model'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from hack_model import load_identity_basis
assert load_identity_basis().mean.shape == (14062, 3)
assert 'hack_model.core' not in sys.modules
'''
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable, '-I', '-c', code], cwd=cwd, check=True)


if __name__ == '__main__':
    unittest.main()
