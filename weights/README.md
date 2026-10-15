# Pretrained weights

The weights are hosted on Hugging Face, not in this repository:
https://huggingface.co/mri2sff/anon-i2i-t2-to-sff

| File | Model |
|---|---|
| `unet_baseline.pth` | no augmentation |
| `unet_augmented.pth` | generic nnU-Net-style augmentation |

Download both files into this folder, so that it looks like this:

```
weights/
├── unet_baseline.pth
└── unet_augmented.pth
```

Either download them manually from the "Files and versions" tab, or run this
from the repository root (requires `pip install huggingface_hub`):

```python
from huggingface_hub import hf_hub_download

for name in ["unet_baseline.pth", "unet_augmented.pth"]:
    hf_hub_download(repo_id="mri2sff/anon-i2i-t2-to-sff", filename=name, local_dir="weights")
```

See the main [README](../README.md) for usage.
