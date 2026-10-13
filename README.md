# T2w → SFF Image-to-Image Translation (U-Net)

Code and pretrained weights for translating axial **T2-weighted HASTE** MR slices
into synthetic **signal fat fraction (SFF)** maps with a U-Net. Two pretrained
models are provided:

| Model | Weights file | Training |
|---|---|---|
| `baseline` | `weights/unet_baseline.pth` | no augmentation |
| `augmented` | `weights/unet_augmented.pth` | generic nnU-Net-style augmentation (gamma, brightness, noise, blur) on the T2w input |

Both models share the same architecture, hyper-parameters and initialisation
and differ only in the augmentation.

> **Anonymous submission.** Example images (T2w input and synthetic SFF output)
> will be added to this repository later. Until then, synthetic dummy data can be
> generated with `examples/make_dummy_data.py`.

> **Research code, not for clinical use.** The models were trained on a single
> protocol (see "Training data") and are not validated for other scanners or sequences.

## Repository structure

```
├── infer.py                 # inference on a new 2D T2w slice
├── train.py                 # training, with / without augmentation (--augment / --no-augment)
├── environment.yml          # conda environment
├── weights/                 # pretrained weights (state_dict files)
├── src/
│   ├── model.py             # U-Net
│   ├── preprocessing.py     # normalisation + orientation helpers
│   └── augmentations.py     # generic nnU-Net-style T2w augmentation
└── examples/make_dummy_data.py
```

## 1. Installation

```bash
conda env create -f environment.yml
conda activate sff-translation
```

The environment installs PyTorch 2.7.1 built for CUDA 11.8 (the CUDA runtime is
bundled; you only need a recent NVIDIA driver). For CPU-only use, edit
`environment.yml` as described in the comment inside it. Inference also runs fine on CPU.

## 2. Inference on a new image

```bash
python infer.py --input my_slice.npy --model baseline  --orientation standard
python infer.py --input my_slice.npz --key T2 --model augmented --orientation standard
```

| Argument | Meaning |
|---|---|
| `--input` | `.npy` file, or `.npz` file containing the slice |
| `--key` | array name inside a `.npz` (default `T2`; if the `.npz` holds a single array, it is used automatically) |
| `--model` | `baseline` or `augmented` |
| `--orientation` | `standard` or `raw` (**required**, see below) |
| `--weights` | optional path to override the default weights |
| `--output_dir` | default `outputs/` |

Outputs: `<name>_<model>_sff.npy` (float32 SFF map, values in [0, 1]) and a
`.png` preview. The output has the **same orientation as your input**.

### Input requirements

- **A single 2D axial slice**, no volumes. For a volume, loop over slices yourself.
- **Square**, with height = width a **multiple of 16**. **256 × 256 is recommended**;
  the models were trained at this size, and other sizes are not validated.
- Array of any numeric dtype (cast to float32), no NaN/Inf, **magnitude** T2w image.
  The **absolute intensity scale does not matter**: every slice is min-max
  normalised to [0, 1] internally (`src/preprocessing.py`).
- **Orientation**, which must be right (see next section).
- Acquisition similar to the training data (see "Training data"): axial T2w HASTE, ~1.4 mm in-plane.

### Orientation

The models were trained **without** any orientation correction, i.e. on arrays in a
specific "raw" layout. You must therefore tell `infer.py` which layout your array has:

- `--orientation standard` (usual case): axial slice as in most viewers
  (radiological convention): **anterior (abdomen) at the top, posterior (back) at
  the bottom, the patient's right on the left of the image**.
  The script converts to the training layout internally and converts the result
  back, so the SFF map is pixel-aligned with your input.
- `--orientation raw`: your array already has the training layout, which is the
  standard layout **transposed and flipped**: `raw = standard[::-1, ::-1].T`.
  In this layout, the left–right axis runs along the rows and the
  anterior–posterior axis along the columns.

Quick check of your own data: display `raw_to_standard(arr)` (from `src/preprocessing.py`)
with `plt.imshow`. If the abdomen is at the top and the patient's right is on the
image left, `raw` is correct for that array.

## 3. Training

### Training data format

`--train_dir` and `--val_dir` each contain one `.npz` file **per 2D slice** with two arrays:

| Array name (default) | Content |
|---|---|
| `T2` | raw T2w slice, shape `(256, 256)`, any intensity scale, **raw layout** (see above) |
| `SFF` | SFF map, same shape and layout, float values in **[0, 1]** (not normalised by the code) |

The array names can be changed with `--input_key` / `--target_key`. Input and target
must be pixel-aligned. The T2w slice is min-max normalised inside the training code.
Do not normalise anything beforehand.

### Commands

```bash
# without augmentation
python train.py --train_dir data/train --val_dir data/val --out_dir runs/baseline  --no-augment

# with augmentation (nnU-Net-style: gamma, brightness, noise, blur; each slice augmented with p = 0.7)
python train.py --train_dir data/train --val_dir data/val --out_dir runs/augmented --augment
```

Settings (both models): AdamW, lr 2e-4, batch size 16, L1 loss, up to 200 epochs,
validation SSIM every 5 epochs on a fixed 500-slice subset, ReduceLROnPlateau,
early stopping (patience 10, 5 warm-up epochs). With `--augment`, the validation
subset is augmented once with a fixed seed and then kept frozen. The best checkpoint
is saved as `best.pth`, the last as `final.pth`; both are plain `state_dict` files
usable with `infer.py --weights`.

### Smoke test without real data

```bash
python examples/make_dummy_data.py
python train.py --train_dir examples/dummy_data/train --val_dir examples/dummy_data/val \
                --out_dir runs/dummy --augment --epochs 5 --num_workers 0
python infer.py --input examples/dummy_data/dummy_input.npy --model baseline \
                --weights runs/dummy/final.pth --orientation raw
```

The dummy images are random ellipses; the resulting model is meaningless.

## Training data

The pretrained weights were trained on 2D slices from a large population-based
cohort study (name withheld for anonymous review); the data are **not** part of this
repository. Acquisition: 3.0 T Siemens MAGNETOM Skyra.
- T2-weighted HASTE (input): slice thickness 5.0 mm, in-plane resolution 1.4 × 1.4 mm, TE 81 ms, coverage from shoulder to epigastric region.
- T1-weighted 3D VIBE two-point Dixon (source of the SFF target maps): slice thickness 3.0 mm, in-plane resolution 1.4 × 1.4 mm, coverage neck to knee, CAIPIRINHA acceleration.

## License

Code: MIT (see `LICENSE`). Use of the pretrained weights may be subject to the terms of the data
source they were trained on.
