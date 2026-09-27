# HACK
<img src="figs/teaser.jpg" width="100%">

HACK (Head-And-neCK) is a novel parametric model for constructing the head and cervical region of digital humans. It aims to disentangle the full spectrum of neck and larynx motions, detailed facial expressions as well as appearance variations, offering more personalized and anatomically consistent controls that are compatible with CG engines. 


| Feature                | Ready              |
| ---------------------- | ------------------ |
| Template mesh          | :heavy_check_mark: |
| Skeleton joints        | :heavy_check_mark: |
| Skinning weights       | :heavy_check_mark: |
| Larynx operation       | :heavy_check_mark: |
| Shape blendshapes      | :heavy_check_mark: |
| Expression blendshapes | :heavy_check_mark: |
| Pose blendshapes       | :heavy_check_mark: |
| Blendshapes space      | :soon:             |
| PyTorch module         | :heavy_check_mark: |
| Texture basis          | :soon:             |

## Installation and package API

Install the asset package from this fork (no PyTorch required):

```bash
python -m pip install "hack-model @ git+https://github.com/nmfisher/HACK-Model.git"
```

For the PyTorch model, install the `model` extra. For development from a local
checkout, use an editable install:

```bash
python -m pip install -e '.[model]'
# For the notebook, use: python -m pip install -e '.[examples]'
```

The wheel includes all files under `model/`. Imports and model construction
work from any working directory and do not require `metahuman_automation`.
Consumers should locate assets through the package API:

```python
from hack_model import get_asset_root, get_model_dir

model_dir = get_model_dir()
shape_path = model_dir / "S.npy"
template_path = model_dir / "000_generic_neutral_mesh.obj"
# For existing consumers expecting a directory containing model/:
hack_dir = get_asset_root()
```

Only NumPy is required for asset consumers. The PyTorch API loads its optional
dependencies when first accessed:

```python
from hack_model import HACK, load_pca, get_model_dir

hack = HACK()
shape = load_pca(get_model_dir() / "S.npy")
```

`from hack_model import helper` provides the upstream OBJ reader/writer. The
reader preserves vertex order, polygon topology, and per-corner UV indices.
There are no imports from the consuming application or `sys.path` changes.

Applications can declare `hack-model @ git+https://github.com/nmfisher/HACK-Model.git@<commit>`
as a dependency, replacing `<commit>` with a published commit for reproducible
installs. Install the base package into Blender's Python for asset-only use.
Existing `--hack-dir` overrides can continue to point at upstream checkouts;
use `get_asset_root()` for the installed package's default location.

To run the standalone tests after installation:

```bash
python -m unittest discover -s tests -v
```

## HACK (Head-And-neCK) Model
The HACK Model consists of a base topology along with definitions of facial landmarks, shape blendshapes, expression blendshapes, cervical joints, and pose blendshapes.

### Face Model Topology
HACK uses the same facial topology as [ICT-FaceKit](https://github.com/ICT-VGL/ICT-FaceKit), including 14062 vertices and 14034 quad faces.

| Ordinal# | Geometry name    | Vertex range         | Polygon range        | #Vertices | #Faces |
| -------- | ---------------- | -------------------- | -------------------- | --------- | ------ |
| n/a      | All              | `range(0,14062)    ` | `range(0,26384)    ` | 26719     | 26384  |
| #0       | Face             | `range(0,9409)     ` | `range(0,9230)     ` | 9409      | 9230   |
| #1       | Head and Neck    | `range(9409,11248) ` | `range(9230,11144) ` | 1839      | 1914   |
| #2       | Mouth socket     | `range(11248,13294)` | `range(11144,13226)` | 2046      | 2082   |
| #3       | Eye socket left  | `range(13294,13678)` | `range(13226,13630)` | 384       | 404    |
| #4       | Eye socket right | `range(13678,14062)` | `range(13630,14034)` | 384       | 404    |

### Face Area Details
| Ordinal# | Geometry name    | Vertex range    | Polygon range   | #Vertices | #Faces |
| -------- | ---------------- | --------------- | --------------- | --------- | ------ |
| #0       | Full face area   | `range(0,9409)` | `range(0,9230)` | 9409      | 9230   |
| #1       | Narrow face area | `range(0,6706)` | `range(0,6560)` | 6706      | 6560   |



### Facial Landmarks
HACK shares the same Multi-PIE 68 point facial landmarks indices as [ICT-FaceKit](https://github.com/ICT-VGL/ICT-FaceKit):
`[1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966]`


### Expression blendshapes
HACK Model includes 55 expression blendshapes, which have the same definition from [ICT-FaceKit](https://github.com/ICT-VGL/ICT-FaceKit).
> Current expression shapes adopt the naming convention of the Apple ARKit, but with "Left" and "Right" specified with "_L" and "_R". Additionally, we separeate the shapes (browInnerUp_L and browInnerUp_R), and (cheekPuff_L and cheekPuff_R).


### Shape blendshapes
HACK Model includes a set of 200 PCA blendshapes of HACK shape space. 

<img src="figs/S_evaluation.png" width="50%">


### Joints
HACK Model includes 8 joints corresponding to the bottom points of 7 vertebrae (C1-C7) and the apex of C1. This setup yields 8 bone transformations denoted as c7-t1, c6-c7, c5-c6, c4-c5, c3-c4, c2-c3, c1-c2,
and o-c1.

<img src="figs/limit.png" width="50%">

### Skinning weights
HACK Model includes skinning weights corresponding to 8 rotation joints in the cervical spine.
<img src="figs/skinning.jpg" width="100%">


### Pose blendshapes
HACK **Model** includes 72 pose blendshapes, whose weights are automatically set by the rotation of 8 joints.

## File Structure
```
HACK-Model
|-- data    # recorded data for demonstration
|-- model
|   |-- 000_generic_neutral_mesh_newuv.obj  # template mesh for larynx
|   |-- 000_generic_neutral_mesh.obj        # template mesh
|   |-- blendshape.npy                      # ICT-FaceKit expressions
|   |-- bones_neutral.json                  # template bone
|   |-- E.npy                               # expression blendshapes
|   |-- Lc_mid.png                          # template larynx shape
|   |-- P.npy                               # pose blendshapes
|   |-- S.npy                               # shape blendshapes
|   |-- ts_larynx.npy                       # template larynx shape
|   |-- weight_map_smooth.npy               # skinning weight
|-- examples.ipynb  # examples
|-- __init__.py     # public package API (installed as hack_model)
|-- assets.py       # working-directory-independent asset paths
|-- identity.py     # NumPy-only identity basis and triangulated topology
|-- core.py         # optional PyTorch HACK model
|-- helper.py       # OBJ reader/writer
|-- pyproject.toml  # package metadata and dependencies
```

## Examples

### PyTorch model
Install `.[examples]`, then open `examples.ipynb` from this checkout for running examples.

### NumPy identity geometry (including Blender)

```python
import numpy as np
from hack_model import load_identity_basis

basis = load_identity_basis()  # packaged assets; optional asset_root contains model/
coefficients = np.zeros(len(basis.vt_std))
vertices = basis.mean + np.einsum("k,kvc->vc", coefficients, basis.vt_std)
triangles = basis.faces
```

This API needs only NumPy, not Torch, OpenCV, Roma or Blender. It returns
float64 `mean` `(V,3)`, float64 `vt_std` `(K,V,3)` and int32 triangles `(F,3)`
in native coordinates. Vertex IDs, PCA scale and triangle order are preserved;
quads are split across the 0–2 diagonal. It does not apply pose, expression,
normalization or identity sampling. Only load trusted assets: upstream `S.npy`
contains a pickled dictionary. Malformed shapes, non-finite arrays and
invalid topology raise `ValueError` rather than silently changing geometry.


### Blender model
Coming soon...

<!-- ## Publications -->

## Contact
If you have any questions, please send an e-mail to shanghaitechmars@foxmail.com
