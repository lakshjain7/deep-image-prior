"""Image quality metrics. Every metric compares an estimate `sr` against the original `hr`.

Images: numpy (H, W, 3), float32, RGB, values in [0, 1], same shape.

Two PSNR versions:
  psnr_rgb - all three colour channels, whole image (what the training curves use).
  psnr_y   - brightness (Y) channel only, with a border of `shave` pixels removed on every side.
             This is the usual super-resolution protocol and is what the DIP paper's table uses
             (checked against the paper's Bicubic row in repro/checks.py).
"""
import numpy as np


def psnr(a, b, peak=1.0):
    """PSNR in dB between two arrays of the same shape: 10 * log10(peak^2 / mean((a - b)^2)).

    Return float("inf") if the arrays are identical.
    """
    raise NotImplementedError("M2: write psnr")


def psnr_rgb(hr, sr):
    """PSNR over all pixels and all three channels."""
    raise NotImplementedError("M2: write psnr_rgb")


def rgb_to_y(img):
    """Brightness channel, MATLAB convention (ITU-R BT.601), result in [16/255, 235/255].

        Y = (16 + 65.481 * R + 128.553 * G + 24.966 * B) / 255      with R, G, B in [0, 1]

    Return shape (H, W).
    """
    raise NotImplementedError("M2: write rgb_to_y")


def psnr_y(hr, sr, shave=4):
    """PSNR on the Y channel after removing `shave` pixels from each border (shave = scale factor)."""
    raise NotImplementedError("M2: write psnr_y")
