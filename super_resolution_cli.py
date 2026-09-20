#!/usr/bin/env python3
"""
Deep Image Prior: Super-Resolution Pipeline
Refactored CLI script with support for CPU/CUDA fallback, automated logging, and W&B integration.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless terminal execution
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import torch
import torch.optim

from models import get_net
from models.downsampler import Downsampler
from utils.sr_utils import (
    get_baselines,
    get_noise,
    get_params,
    load_LR_HR_imgs_sr,
    np_to_torch,
    optimize,
    plot_image_grid,
    put_in_center,
    torch_to_np,
    tv_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Image Prior Super-Resolution")
    parser.add_argument(
    "--wandb-entity",
    type=str,
    default="deepimageprior",
    help="Weights & Biases team/entity name",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        default="data/sr/zebra_GT.png",
        help="Path to the high-resolution ground truth image",
    )
    parser.add_argument(
        "--factor",
        type=int,
        choices=[4, 8],
        default=4,
        help="Super-resolution scaling factor (4 or 8)",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help="Number of optimization iterations (defaults to 2000 for x4, 4000 for x8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--reg-noise-std",
        type=float,
        default=None,
        help="Input noise perturbation std (defaults to 0.03 for x4, 0.05 for x8)",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.0,
        help="Total Variation regularization weight",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sr",
        help="Directory to save reconstructed images and metric logs",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=100,
        help="Frequency (in iterations) to log metrics and save preview images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Hardware execution device",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="deep-image-prior",
        help="Weights & Biases project name",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        chosen = device_arg

    if chosen == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA was requested but is not available. Falling back to CPU.")
        chosen = "cpu"

    dtype = torch.cuda.FloatTensor if chosen == "cuda" else torch.FloatTensor
    print(f"[*] Running on device: {chosen.upper()}")
    return chosen, dtype


def main():
    args = parse_args()
    device, dtype = resolve_device(args.device)

    if device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Setup directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve default hyperparameters based on scaling factor
    if args.factor == 4:
        num_iter = args.num_iter or 2000
        reg_noise_std = args.reg_noise_std if args.reg_noise_std is not None else 0.03
    elif args.factor == 8:
        num_iter = args.num_iter or 4000
        reg_noise_std = args.reg_noise_std if args.reg_noise_std is not None else 0.05
    else:
        raise ValueError(f"Unsupported factor: {args.factor}")

    # Optional W&B setup
    wb_run = None
    if args.wandb:
        try:
            import wandb

            wb_run = wandb.init(
                        entity=args.wandb_entity,
                        project=args.wandb_project,
                        config={
                            "image": args.img_path,
                            "factor": args.factor,
                            "num_iter": num_iter,
                            "lr": args.lr,
                            "reg_noise_std": reg_noise_std,
                            "tv_weight": args.tv_weight,
                            "device": device,
                        },
                    )
            print("[*] Initialized Weights & Biases logging.")
        except ImportError:
            print("[Warning] wandb is not installed. Skipping wandb tracking. Run 'uv pip install wandb' to enable.")

    # 1. Load images and baselines
    print(f"[*] Loading input image: {args.img_path}")
    imgs = load_LR_HR_imgs_sr(args.img_path, imsize=-1, factor=args.factor, enforse_div32="CROP")
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(imgs["LR_pil"], imgs["HR_pil"])

    psnr_bicubic = compare_psnr(imgs["HR_np"], imgs["bicubic_np"])
    psnr_nearest = compare_psnr(imgs["HR_np"], imgs["nearest_np"])
    print(f"[*] Baseline PSNR - Bicubic: {psnr_bicubic:.4f} dB | Nearest: {psnr_nearest:.4f} dB")

    # 2. Build model and input noise tensor
    input_depth = 32
    INPUT = "noise"
    pad = "reflection"
    OPT_OVER = "net"
    KERNEL_TYPE = "lanczos2"
    OPTIMIZER = "adam"

    net_input = get_noise(input_depth, INPUT, (imgs["HR_pil"].size[1], imgs["HR_pil"].size[0])).type(dtype).detach()
    net = get_net(
        input_depth,
        "skip",
        pad,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode="bilinear",
    ).type(dtype)

    mse = torch.nn.MSELoss().type(dtype)
    img_LR_var = np_to_torch(imgs["LR_np"]).type(dtype)
    downsampler = Downsampler(n_planes=3, factor=args.factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

    # 3. Optimization Setup
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    psnr_history = []
    iteration_counter = {"i": 0}

    def closure():
        nonlocal net_input
        i = iteration_counter["i"]

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        net.zero_grad()
        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var)
        if args.tv_weight > 0:
            total_loss += args.tv_weight * tv_loss(out_HR)

        total_loss.backward()

        # Compute Metrics
        out_HR_np = np.clip(torch_to_np(out_HR), 0, 1)
        out_LR_np = np.clip(torch_to_np(out_LR), 0, 1)

        psnr_LR = compare_psnr(imgs["LR_np"], out_LR_np)
        psnr_HR = compare_psnr(imgs["HR_np"], out_HR_np)
        psnr_history.append((i, psnr_LR, psnr_HR, total_loss.item()))

        # Live terminal progress
        sys.stdout.write(f"\r[Iteration {i:05d}/{num_iter:05d}] Loss: {total_loss.item():.6f} | PSNR_LR: {psnr_LR:.2f} dB | PSNR_HR: {psnr_HR:.2f} dB")
        sys.stdout.flush()

        # Checkpoints & Previews
        if i % args.log_freq == 0 or i == num_iter - 1:
            preview_img = (out_HR_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(preview_img).save(output_path / f"iter_{i:05d}.png")

            if wb_run:
                import wandb
                wb_run.log({
                    "iteration": i,
                    "loss": total_loss.item(),
                    "psnr_LR": psnr_LR,
                    "psnr_HR": psnr_HR,
                    "reconstructed_HR": wandb.Image(preview_img),
                })

        iteration_counter["i"] += 1
        return total_loss

    print(f"[*] Starting optimization for {num_iter} iterations...")
    params = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, params, closure, args.lr, num_iter)
    print("\n[*] Optimization completed.")

    # 4. Final Image Generation and Saving
    with torch.no_grad():
        final_HR_np = np.clip(torch_to_np(net(net_input_saved)), 0, 1)
        final_deep_prior = put_in_center(final_HR_np, imgs["orig_np"].shape[1:])

    final_psnr_hr = compare_psnr(imgs["HR_np"], final_HR_np)
    print(f"[*] Final Super-Resolved Image PSNR: {final_psnr_hr:.4f} dB")

    final_img_arr = (final_HR_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    final_output_file = output_path / f"final_sr_x{args.factor}.png"
    Image.fromarray(final_img_arr).save(final_output_file)
    print(f"[*] Saved final high-resolution image to: {final_output_file}")

    # Save PSNR curve to CSV
    metrics_csv = output_path / "psnr_history.csv"
    np.savetxt(metrics_csv, np.array(psnr_history), delimiter=",", header="iteration,psnr_lr,psnr_hr,loss", comments="")
    print(f"[*] Saved training metrics log to: {metrics_csv}")

    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    main()