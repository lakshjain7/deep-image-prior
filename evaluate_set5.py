
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)

    im_ycrcb = cv2.cvtColor(
        im_rgb,
        cv2.COLOR_RGB2YCR_CB
    )

    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)

    im_ycbcr[:, :, 0] = (
        im_ycbcr[:, :, 0] * (235 - 16) + 16
    ) / 255.0

    im_ycbcr[:, :, 1:] = (
        im_ycbcr[:, :, 1:] * (240 - 16) + 16
    ) / 255.0

    return im_ycbcr


def compare_psnr_y(x, y):
    return compare_psnr(
        rgb2ycbcr(x.transpose(1, 2, 0))[:, :, 0],
        rgb2ycbcr(y.transpose(1, 2, 0))[:, :, 0]
    )


def center_crop_div32(img):
    """
    Reproduce the CROP behavior of load_LR_HR_imgs_sr().
    img shape: C x H x W
    """

    c, h, w = img.shape

    new_h = h - (h % 32)
    new_w = w - (w % 32)

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    return img[
        :,
        top:top + new_h,
        left:left + new_w
    ]


def load_image(path):
    img = np.array(
        Image.open(path).convert("RGB")
    ).astype(np.float32) / 255.0

    return img.transpose(2, 0, 1)


base = "/content/deep-image-prior"

data = f"{base}/data/sr/Set5/Set5"
outputs = f"{base}/outputs/sr/set5_x4"

names = [
    "baby",
    "bird",
    "butterfly",
    "head",
    "woman",
]

paper = {
    "baby": 31.49,
    "bird": 31.80,
    "butterfly": 26.23,
    "head": 31.04,
    "woman": 28.93,
}

results = []


for name in names:

    gt_path = f"{data}/{name}.png"

    if name == "baby":
        sr_path = f"{outputs}/final_sr_x4.png"
    else:
        sr_path = f"{outputs}/{name}/final_sr_x4.png"

    gt = load_image(gt_path)
    sr = load_image(sr_path)

    original_shape = gt.shape

    # IMPORTANT:
    # This reproduces enforse_div32="CROP"
    gt = center_crop_div32(gt)

    print(
        f"{name:10s} | "
        f"Original GT: {original_shape[1:]} | "
        f"Cropped GT: {gt.shape[1:]} | "
        f"SR: {sr.shape[1:]}"
    )

    # The generated SR should now have exactly the same dimensions.
    assert gt.shape == sr.shape, (
        f"Shape mismatch for {name}: "
        f"GT={gt.shape}, SR={sr.shape}"
    )

    # Same 4-pixel border exclusion used in our evaluator.
    h = gt.shape[1]
    w = gt.shape[2]

    gt_eval = gt[:3, 4:h-4, 4:w-4]
    sr_eval = sr[:3, 4:h-4, 4:w-4]

    score = compare_psnr_y(gt_eval, sr_eval)

    results.append(score)

    print(
        f"{'':10s} | "
        f"Ours: {score:.4f} dB | "
        f"Paper: {paper[name]:.2f} dB | "
        f"Difference: {score - paper[name]:+.4f} dB"
    )


average = np.mean(results)

print("\n" + "=" * 60)
print(f"Ours average : {average:.4f} dB")
print(f"Paper average: 29.89 dB")
print(f"Difference   : {average - 29.89:+.4f} dB")
print("=" * 60)
