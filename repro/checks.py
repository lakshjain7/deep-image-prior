"""Checks for the repro/ milestones. Each check prints PASS/FAIL lines and returns True if all passed.

These are tests of YOUR code in repro/data.py and repro/metrics.py against known values:
  - Set5 image sizes,
  - the bicubic-baseline PSNR our earlier pipeline recorded for baby (30.73 dB),
  - independent PSNR implementations (scikit-image, evaluate_set5.py),
  - the DIP paper's Bicubic row (Ulyanov et al. 2017, 4x super-resolution table).
"""
import numpy as np

from repro import data, metrics

HR_SIZES = {"baby": (512, 512), "bird": (288, 288), "butterfly": (256, 256), "head": (280, 280), "woman": (344, 228)}
CROPPED_SIZES = {"baby": (512, 512), "bird": (288, 288), "butterfly": (256, 256), "head": (256, 256), "woman": (320, 224)}
PAPER_BICUBIC = {"baby": 31.78, "bird": 30.20, "butterfly": 22.13, "head": 31.34, "woman": 26.75}
PAPER_DIP = {"baby": 31.49, "bird": 31.80, "butterfly": 26.23, "head": 31.04, "woman": 28.93}


def _report(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    return ok


def _ref_psnr(a, b):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(a, b, data_range=1.0)


def _on_8bit_grid(img):
    return np.allclose(img * 255, np.round(img * 255), atol=1e-3)


def check_m1(set5_root):
    """M1: repro/data.py."""
    results = []
    paths = data.find_set5(set5_root)
    results.append(_report(sorted(paths) == sorted(data.SET5_NAMES), f"find_set5 found {sorted(paths)}"))
    for name in data.SET5_NAMES:
        hr = data.load_hr(paths[name])
        ok = hr.shape == HR_SIZES[name] + (3,) and hr.dtype == np.float32 and 0 <= hr.min() and hr.max() <= 1
        results.append(_report(ok, f"load_hr({name}): shape {hr.shape}, dtype {hr.dtype}, range [{hr.min():.2f}, {hr.max():.2f}]"))
        c = data.crop_to_multiple(hr, 32)
        results.append(_report(c.shape[:2] == CROPPED_SIZES[name], f"crop_to_multiple({name}): {c.shape[:2]} (expected {CROPPED_SIZES[name]})"))
    hr = data.crop_to_multiple(data.load_hr(paths["baby"]), 32)
    lr = data.downsample(hr, 4, "pil-lanczos")
    results.append(_report(lr.shape == (128, 128, 3), f"downsample(baby): shape {lr.shape} (expected (128, 128, 3))"))
    results.append(_report(lr.dtype == np.float32 and _on_8bit_grid(lr), "downsample output is float32 on the 8-bit grid"))
    up = data.upsample_bicubic(lr, hr.shape[:2])
    results.append(_report(up.shape == hr.shape, f"upsample_bicubic(baby): shape {up.shape}"))
    value = _ref_psnr(hr, up)
    results.append(_report(abs(value - 30.73) < 0.02,
                           f"bicubic baseline PSNR(baby, Lanczos LR) = {value:.3f} dB (earlier pipeline: 30.73)"))
    q = data.to_uint8_and_back(np.array([[[0.0, 0.5, 1.2]]], dtype=np.float32))
    results.append(_report(np.allclose(q, [[[0.0, 128 / 255, 1.0]]]), f"to_uint8_and_back -> {q.ravel()}"))
    print(f"\nM1: {sum(results)}/{len(results)} checks passed")
    return all(results)


def check_m2(set5_root):
    """M2: repro/metrics.py, then the paper's Bicubic row."""
    from evaluate_set5 import calculate_psnr_y
    results = []
    rng = np.random.default_rng(0)
    a = rng.random((64, 48, 3), dtype=np.float32)
    b = np.clip(a + rng.normal(0, 0.05, a.shape).astype(np.float32), 0, 1)
    results.append(_report(abs(metrics.psnr_rgb(a, b) - _ref_psnr(a, b)) < 1e-4,
                           f"psnr_rgb = {metrics.psnr_rgb(a, b):.4f} vs scikit-image {_ref_psnr(a, b):.4f}"))
    results.append(_report(metrics.psnr_rgb(a, a) == float("inf"), "psnr of identical images is inf"))
    y = metrics.rgb_to_y(a)
    results.append(_report(y.shape == a.shape[:2] and 16 / 255 - 1e-6 <= y.min() and y.max() <= 235 / 255 + 1e-6,
                           f"rgb_to_y: shape {y.shape}, range [{y.min():.4f}, {y.max():.4f}] (expected within [0.0627, 0.9216])"))
    mine, ref = metrics.psnr_y(a, b, 4), calculate_psnr_y(a[4:-4, 4:-4], b[4:-4, 4:-4])
    results.append(_report(abs(mine - ref) < 1e-3, f"psnr_y = {mine:.4f} vs evaluate_set5.py {ref:.4f}"))

    print("\nPaper's Bicubic row (HR -> PIL bicubic down x4 -> PIL bicubic up, Y channel, 4-px border):")
    paths = data.find_set5(set5_root)
    print(f"{'image':10s} {'paper':>7s} {'yours':>7s} {'diff':>7s}")
    diffs = []
    for name in data.SET5_NAMES:
        hr = data.load_hr(paths[name])
        up = data.upsample_bicubic(data.downsample(hr, 4, "pil-bicubic"), hr.shape[:2])
        v = metrics.psnr_y(hr, up, 4)
        diffs.append(v - PAPER_BICUBIC[name])
        print(f"{name:10s} {PAPER_BICUBIC[name]:7.2f} {v:7.2f} {v - PAPER_BICUBIC[name]:+7.2f}")
    mean_paper = np.mean(list(PAPER_BICUBIC.values()))
    print(f"{'mean':10s} {mean_paper:7.2f} {mean_paper + np.mean(diffs):7.2f} {np.mean(diffs):+7.2f}")
    results.append(_report(max(abs(d) for d in diffs) < 0.15, "every image within 0.15 dB of the paper's Bicubic row "
                           "(small gaps can come from PIL vs MATLAB resizing; ~1 dB everywhere means the metric is wrong)"))
    print(f"\nM2: {sum(results)}/{len(results)} checks passed")
    return all(results)
