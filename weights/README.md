# Pretrained weights

The pretrained weights are **not stored in this GitHub repository**. They are hosted
on Hugging Face and have to be downloaded into this folder.

| File | Model | Training |
|---|---|---|
| `unet_baseline.pth` | `baseline` | no augmentation |
| `unet_augmented.pth` | `augmented` | generic nnU-Net-style augmentation on the T2w input |

Hugging Face repository: <HF_REPO_URL>

## Required file layout

After downloading, the folder must look exactly like this (the names are what
`infer.py` expects by default):

```
weights/
├── README.md
├── unet_baseline.pth
└── unet_augmented.pth
```

## Download

**Option A: browser.** Open the Hugging Face repository, go to "Files and versions",
and download both `.pth` files (download icon next to each file). Move them into this folder.

**Option B: command line / Python** (requires `pip install huggingface_hub`):

```python
from huggingface_hub import hf_hub_download

for name in ["unet_baseline.pth", "unet_augmented.pth"]:
    hf_hub_download(repo_id="<HF_USER>/<HF_REPO>", filename=name, local_dir="weights")
```

Run this from the root of the repository.

**Option C: direct links** (e.g. with `wget` or `curl -L -O`):

```
<HF_REPO_URL>/resolve/main/unet_baseline.pth
<HF_REPO_URL>/resolve/main/unet_augmented.pth
```

## Verify the download (optional)

SHA-256 checksums:

```
<SHA256_BASELINE>  unet_baseline.pth
<SHA256_AUGMENTED> unet_augmented.pth
```

Linux: `sha256sum weights/*.pth`, macOS: `shasum -a 256 weights/*.pth`,
Windows: `certutil -hashfile weights\unet_baseline.pth SHA256`.

## File format

Both files are plain PyTorch `state_dict` files (tensors only, no pickled code) for the
`UNet` class in `src/model.py`. Input is a min-max normalised 2D T2w slice of shape
`(1, 1, H, W)` with H = W a multiple of 16 (256 recommended); output is an SFF map in [0, 1].

```python
import torch
from src.model import UNet

model = UNet()
model.load_state_dict(torch.load("weights/unet_baseline.pth", map_location="cpu", weights_only=True))
model.eval()
```

## Usage

See the main [README](../README.md) for inference (`infer.py`), the required input
format and orientation, and training (`train.py`).

Use of the weights may be subject to the terms of the data source they were trained on.
