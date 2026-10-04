
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio


IMAGE_NAMES = [
    "baby",
    "bird",
    "butterfly",
    "head",
    "woman",
]

# Table 2: DIP paper, Set5, x4
PAPER_PSNR = {
    "baby": 31.49,
    "bird": 31.80,
    "butterfly": 26.23,
    "head": 31.04,
    "woman": 28.93,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DIP x4 super-resolution results on Set5."
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing the Set5 images.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory containing generated SR outputs.",
    )

    return parser.parse_args()


def find_set5_images(dataset_root):
    """
    Search recursively under dataset_root for the five Set5 images.

    This avoids assuming a particular directory nesting such as:
        Set5/Set5/
        Set5/set5/Set5/
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_root}"
        )

    images = {}

    for name in IMAGE_NAMES:
        matches = list(dataset_root.rglob(f"{name}.png"))

        if len(matches) == 0:
            raise FileNotFoundError(
                f"Could not find {name}.png under {dataset_root}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Found multiple copies of {name}.png under "
                f"{dataset_root}:\n"
                + "\n".join(str(path) for path in matches)
            )

        images[name] = matches[0]

    return images


def crop_to_div32(image):
    """
    Reproduce the center crop used by load_LR_HR_imgs_sr()
    when enforse_div32='CROP'.
    """

    height, width = image.shape[:2]

    new_height = height - height % 32
    new_width = width - width % 32

    top = (height - new_height) // 2
    left = (width - new_width) // 2

    return image[
        top:top + new_height,
        left:left + new_width
    ]


def rgb2ycbcr(image_rgb):
    """
    Convert RGB image in [0, 1] to the YCbCr representation
    used by the original DIP evaluation code.
    """

    image_rgb = image_rgb.astype(np.float32)

    image_ycrcb = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2YCR_CB
    )

    image_ycbcr = image_ycrcb[:, :, (0, 2, 1)].astype(np.float32)

    image_ycbcr[:, :, 0] = (
        image_ycbcr[:, :, 0] * (235 - 16) + 16
    ) / 255.0

    image_ycbcr[:, :, 1:] = (
        image_ycbcr[:, :, 1:] * (240 - 16) + 16
    ) / 255.0

    return image_ycbcr


def calculate_psnr_y(gt, sr):
    """
    Calculate PSNR using only the Y/luminance channel.
    """

    gt_y = rgb2ycbcr(gt)[:, :, 0]
    sr_y = rgb2ycbcr(sr)[:, :, 0]

    return peak_signal_noise_ratio(
        gt_y,
        sr_y,
        data_range=1.0
    )


def load_image(path):
    """
    Load an image as RGB float32 in the range [0, 1].
    """

    return (
        np.asarray(
            Image.open(path).convert("RGB"),
            dtype=np.float32,
        )
        / 255.0
    )


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    print("=" * 75)
    print("Set5 x4 Super-Resolution Evaluation")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Find the five Set5 ground-truth images dynamically
    # ---------------------------------------------------------

    gt_images = find_set5_images(dataset_root)

    print("\nGround-truth images:")
    for name in IMAGE_NAMES:
        print(f"  {name:10s} -> {gt_images[name]}")

    print("\nGenerated SR images:")
    for name in IMAGE_NAMES:
        sr_path = output_root / name / "final_sr_x4.png"
        print(f"  {name:10s} -> {sr_path}")

    # ---------------------------------------------------------
    # 2. Evaluate each image
    # ---------------------------------------------------------

    results = []

    print("\n" + "=" * 75)
    print("Per-image results")
    print("=" * 75)

    for name in IMAGE_NAMES:

        gt_path = gt_images[name]
        sr_path = output_root / name / "final_sr_x4.png"

        if not sr_path.exists():
            raise FileNotFoundError(
                f"Generated SR image not found for {name}:\n"
                f"{sr_path}"
            )

        # Load GT and generated SR image
        gt = load_image(gt_path)
        sr = load_image(sr_path)

        original_gt_shape = gt.shape[:2]
        original_sr_shape = sr.shape[:2]

        # -----------------------------------------------------
        # Match the preprocessing used by DIP
        # -----------------------------------------------------

        gt = crop_to_div32(gt)

        cropped_gt_shape = gt.shape[:2]

        # The generated SR image should already have the same
        # dimensions as the cropped GT image.
        if gt.shape[:2] != sr.shape[:2]:
            raise RuntimeError(
                f"Dimension mismatch for {name}:\n"
                f"  Cropped GT: {gt.shape[:2]}\n"
                f"  SR:         {sr.shape[:2]}"
            )

        # -----------------------------------------------------
        # Remove 4-pixel border
        # -----------------------------------------------------

        gt_eval = gt[4:-4, 4:-4]
        sr_eval = sr[4:-4, 4:-4]

        # -----------------------------------------------------
        # Calculate Y-channel PSNR
        # -----------------------------------------------------

        psnr = calculate_psnr_y(
            gt_eval,
            sr_eval
        )

        paper_value = PAPER_PSNR[name]
        difference = psnr - paper_value

        results.append(psnr)

        print(
            f"{name:10s} | "
            f"Original GT: {original_gt_shape} | "
            f"Cropped GT: {cropped_gt_shape} | "
            f"SR: {original_sr_shape} | "
            f"Ours: {psnr:.4f} dB | "
            f"Paper: {paper_value:.2f} dB | "
            f"Diff: {difference:+.4f} dB"
        )

    # ---------------------------------------------------------
    # 3. Calculate Set5 average
    # ---------------------------------------------------------

    ours_average = float(np.mean(results))
    paper_average = float(
        np.mean(
            [PAPER_PSNR[name] for name in IMAGE_NAMES]
        )
    )

    difference = ours_average - paper_average

    print("\n" + "=" * 75)
    print("Set5 x4 summary")
    print("=" * 75)

    print(f"Ours average : {ours_average:.4f} dB")
    print(f"Paper average: {paper_average:.2f} dB")
    print(f"Difference   : {difference:+.4f} dB")

    print("=" * 75)


if __name__ == "__main__":
    main()
