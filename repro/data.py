"""Data for DIP super-resolution: load Set5 HR images and make the LR inputs.

Conventions used everywhere in repro/:
    images are numpy arrays, shape (H, W, 3), dtype float32, RGB, values in [0, 1].

Reference behaviour (what the DIP authors' code does), in this repository:
    utils/sr_utils.py, load_LR_HR_imgs_sr()  - crop to a multiple of 32, then PIL Lanczos shrink
    utils/sr_utils.py, get_baselines()        - bicubic upsampling of the LR image
"""
from pathlib import Path

import numpy as np
from PIL import Image

SET5_NAMES = ["baby", "bird", "butterfly", "head", "woman"]


def find_set5(root):
    """Return {name: Path} for the five Set5 HR images found anywhere under `root`.

    Match the file stem (e.g. "baby.png" -> "baby"). Raise FileNotFoundError naming the
    missing image if any of the five is not found exactly once.
    """
    raise NotImplementedError("M1: write find_set5")


def load_hr(path):
    """Load an image file as RGB float32 in [0, 1], shape (H, W, 3).

    Hint: Image.open(path).convert("RGB"), then np.asarray(..., dtype=np.float32) / 255.
    """
    raise NotImplementedError("M1: write load_hr")


def crop_to_multiple(img, m=32):
    """Center-crop `img` so height and width are multiples of `m` (DIP needs multiples of 32).

    Same as DIP: new size = size - size % m, and the crop is centered.
    Check: baby 512x512 stays 512x512; head 280x280 becomes 256x256; woman 344x228 (HxW)
    becomes 320x224.
    """
    raise NotImplementedError("M1: write crop_to_multiple")


def to_uint8_and_back(img):
    """Round a float image to 8-bit and back to float, as saving a PNG does.

    Hint: np.clip(img, 0, 1), multiply by 255, np.round, astype(np.uint8), then / 255 as float32.
    """
    raise NotImplementedError("M1: write to_uint8_and_back")


def downsample(hr, factor=4, method="pil-lanczos"):
    """Make the LR input from an HR image: shrink by `factor`, return float32 in [0, 1].

    method:
      "pil-lanczos" - DIP's choice. PIL Image.Resampling.LANCZOS (antialiased).
      "pil-bicubic" - PIL Image.Resampling.BICUBIC (antialiased); used only to check the
                      paper's bicubic row.
    New size = (W // factor, H // factor). Note PIL's resize takes (width, height).
    The result must be 8-bit (go through a PIL uint8 image), as in DIP.
    """
    raise NotImplementedError("M1: write downsample")


def upsample_bicubic(lr, size_hw):
    """Enlarge `lr` to `size_hw` = (H, W) with PIL Image.Resampling.BICUBIC. Return float32 in [0, 1].

    This is the "Bicubic" baseline. Go through a PIL uint8 image, as DIP's get_baselines() does.
    """
    raise NotImplementedError("M1: write upsample_bicubic")
