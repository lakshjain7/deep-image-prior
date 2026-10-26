#!/usr/bin/env python3
"""
Deep Image Prior: Super-Resolution Pipeline
Refactored CLI script with support for CPU/CUDA fallback, automated logging, and W&B integration.
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless terminal execution
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import torch
import torch.optim

from models import get_net
from models.downsampler import Downsampler
from utils.regularizers import OFFICIAL_DEFAULTS, REGULARIZERS, regularization_term
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
    help="Weights & Biases team/entity name (pass an empty string to use your default entity)",
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
        "--kernel-type",
        type=str,
        choices=["lanczos2", "lanczos3"],
        default="lanczos2",
        help="Downsampling kernel used during optimization",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "l1", "l1-em", "m-l1"],
        default="mse",
        help="Fidelity loss: MSE, L1, ensemble-mean L1, or mean multi-kernel L1",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="deep-image-prior",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible runs (default: None)",
    )
    parser.add_argument(
        "--regularizer",
        type=str,
        choices=list(REGULARIZERS),
        default="none",
        help="Paper regularizer (ported from capo-urjc/dip-sisr): none, tv, nn (nuclear norm), "
             "tvssah (SSAHTV), bd (Schiavi backward-diffusion term). Independent of the legacy --tv-weight.",
    )
    parser.add_argument("--reg-weight", type=float, default=0.0, help="Weight of --regularizer")
    parser.add_argument("--p-tv", type=float, default=OFFICIAL_DEFAULTS["p_tv"], help="p of the TV norm")
    parser.add_argument("--p-nn", type=float, default=OFFICIAL_DEFAULTS["p_nn"], help="p of the Schatten norm")
    parser.add_argument("--ssah-mu", type=float, default=OFFICIAL_DEFAULTS["ssah_mu"], help="mu of SSAHTV")
    parser.add_argument("--bd-alpha", type=float, default=OFFICIAL_DEFAULTS["bd_alpha"], help="alpha of the BD term")
    parser.add_argument("--bd-beta", type=float, default=OFFICIAL_DEFAULTS["bd_beta"], help="beta of the BD term")
    parser.add_argument(
        "--lr-method",
        type=str,
        choices=["pil-lanczos", "bicubic"],
        default="pil-lanczos",
        help="How the LR observation is generated from HR: PIL Lanczos (default, Stages 1-3) or the official "
             "dip-sisr bicubic F.interpolate",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Explicit W&B run id")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group")
    parser.add_argument("--wandb-job-type", type=str, default=None, help="W&B job type")
    parser.add_argument("--wandb-tags", type=str, default="", help="Comma-separated W&B tags")
    parser.add_argument(
        "--extra-config-json",
        type=str,
        default="{}",
        help="JSON dict merged into the W&B config (stage name, run key, ...)",
    )
    return parser.parse_args()


def git_info():
    here = Path(__file__).resolve().parent

    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "git_remote": run("git", "config", "--get", "remote.origin.url"),
    }


def env_info(device):
    import skimage

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "skimage": skimage.__version__,
    }
    if device == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def final_metrics(hr_np, sr_np):
    """Metrics of a final reconstruction against the (cropped) ground truth, both CHW float in [0, 1]."""
    out = {
        "psnr_hr": float(compare_psnr(hr_np, sr_np, data_range=1.0)),
        "ssim_hr": float(compare_ssim(hr_np, sr_np, channel_axis=0, data_range=1.0)),
    }
    try:
        # Y-channel PSNR with a 4 px border, same protocol as evaluate_set5.py.
        from evaluate_set5 import calculate_psnr_y

        gt_hwc, sr_hwc = hr_np.transpose(1, 2, 0), sr_np.transpose(1, 2, 0)
        out["psnr_y_border4"] = float(calculate_psnr_y(gt_hwc[4:-4, 4:-4], sr_hwc[4:-4, 4:-4]))
    except ImportError as e:
        print(f"[Warning] Y-channel PSNR skipped: {e}")
        out["psnr_y_border4"] = None
    return out


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
    if args.regularizer != "none" and args.tv_weight > 0:
        raise SystemExit("--tv-weight (legacy TV) cannot be combined with --regularizer; use one of them.")
    if (args.regularizer == "none") != (args.reg_weight == 0):
        raise SystemExit("--regularizer none requires --reg-weight 0, and a regularizer requires a non-zero weight.")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Improve repeatability for supported operations.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[*] Random seed: {args.seed}")
    else:
        # Preserve the existing default behavior.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    device, dtype = resolve_device(args.device)

    if device == "cuda":
        torch.backends.cudnn.enabled = True

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

    config = {
        "image": args.img_path,
        "image_name": Path(args.img_path).stem,
        "factor": args.factor,
        "num_iter": num_iter,
        "lr": args.lr,
        "reg_noise_std": reg_noise_std,
        "tv_weight": args.tv_weight,
        "device": device,
        "kernel_type": args.kernel_type,
        "loss": args.loss,
        "seed": args.seed,
        "regularizer": args.regularizer,
        "reg_weight": args.reg_weight,
        "p_tv": args.p_tv,
        "p_nn": args.p_nn,
        "ssah_mu": args.ssah_mu,
        "bd_alpha": args.bd_alpha,
        "bd_beta": args.bd_beta,
        "optimizer": "adam",
        "lr_method": args.lr_method,
        **git_info(),
        **env_info(device),
        **json.loads(args.extra_config_json),
    }

    # Optional W&B setup
    wb_run = None
    wandb = None
    if args.wandb:
        try:
            import wandb

            wb_run = wandb.init(
                entity=args.wandb_entity or None,
                project=args.wandb_project,
                name=args.wandb_run_name,
                id=args.wandb_run_id,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                tags=[t for t in args.wandb_tags.split(",") if t],
                config=config,
                dir=str(output_path),
            )
            wb_run.define_metric("iteration")
            wb_run.define_metric("*", step_metric="iteration")
            print("[*] Initialized Weights & Biases logging.")
        except ImportError:
            print("[Warning] wandb is not installed. Skipping wandb tracking. Run 'uv pip install wandb' to enable.")

    try:
        run(args, config, device, dtype, num_iter, reg_noise_std, output_path, wb_run, wandb)
    except BaseException as exc:
        (output_path / "error.json").write_text(json.dumps(
            {"error": repr(exc), "traceback": traceback.format_exc(), "config": config}, indent=2, default=str))
        if wb_run:
            wb_run.summary["status"] = "failed"
            wb_run.summary["error"] = repr(exc)
            wb_run.finish(exit_code=1)
        raise
    if wb_run:
        wb_run.finish()


def run(args, config, device, dtype, num_iter, reg_noise_std, output_path, wb_run, wandb):
    start_time = time.time()

    # 1. Load images and baselines
    print(f"[*] Loading input image: {args.img_path}")
    imgs = load_LR_HR_imgs_sr(args.img_path, imsize=-1, factor=args.factor, enforse_div32="CROP",
                              lr_method=args.lr_method)
    imgs["bicubic_np"], imgs["sharp_np"], imgs["nearest_np"] = get_baselines(imgs["LR_pil"], imgs["HR_pil"])

    psnr_bicubic = compare_psnr(imgs["HR_np"], imgs["bicubic_np"], data_range=1.0)
    psnr_nearest = compare_psnr(imgs["HR_np"], imgs["nearest_np"], data_range=1.0)
    print(f"[*] Baseline PSNR - Bicubic: {psnr_bicubic:.4f} dB | Nearest: {psnr_nearest:.4f} dB")

    # 2. Build model and input noise tensor
    input_depth = 32
    INPUT = "noise"
    pad = "reflection"
    OPT_OVER = "net"
    KERNEL_TYPE = args.kernel_type
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

    # Additional operators used only by the multi-kernel loss formulations.
    downsampler_l2 = None
    downsampler_l3 = None
    if args.loss in ("l1-em", "m-l1"):
        downsampler_l2 = Downsampler(
            n_planes=3, factor=args.factor, kernel_type="lanczos2",
            phase=0.5, preserve_size=True
        ).type(dtype)
        downsampler_l3 = Downsampler(
            n_planes=3, factor=args.factor, kernel_type="lanczos3",
            phase=0.5, preserve_size=True
        ).type(dtype)


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

        if args.loss == "mse":
            fidelity = mse(out_LR, img_LR_var)

        elif args.loss == "l1":
            fidelity = torch.nn.functional.l1_loss(out_LR, img_LR_var)

        elif args.loss == "l1-em":
            # L1 against the mean prediction from Lanczos-2 and Lanczos-3.
            pred_l2 = downsampler_l2(out_HR)
            pred_l3 = downsampler_l3(out_HR)
            pred_mean = (pred_l2 + pred_l3) / 2.0
            fidelity = torch.nn.functional.l1_loss(
                pred_mean, img_LR_var
            )

        elif args.loss == "m-l1":
            # Mean of the individual L1 losses, not L1 of the mean prediction.
            pred_l2 = downsampler_l2(out_HR)
            pred_l3 = downsampler_l3(out_HR)
            loss_l2 = torch.nn.functional.l1_loss(pred_l2, img_LR_var)
            loss_l3 = torch.nn.functional.l1_loss(pred_l3, img_LR_var)
            fidelity = (loss_l2 + loss_l3) / 2.0

        reg = regularization_term(
            args.regularizer, out_HR, args.reg_weight, p_tv=args.p_tv, p_nn=args.p_nn,
            ssah_mu=args.ssah_mu, bd_alpha=args.bd_alpha, bd_beta=args.bd_beta,
        )
        if args.tv_weight > 0:
            reg = reg + args.tv_weight * tv_loss(out_HR)  # legacy TV (Stage 2 setting)
        total_loss = fidelity + reg

        total_loss.backward()

        # Compute Metrics
        out_HR_np = np.clip(torch_to_np(out_HR), 0, 1)
        out_LR_np = np.clip(torch_to_np(out_LR), 0, 1)

        psnr_LR = compare_psnr(imgs["LR_np"], out_LR_np, data_range=1.0)
        psnr_HR = compare_psnr(imgs["HR_np"], out_HR_np, data_range=1.0)
        psnr_history.append((i, psnr_LR, psnr_HR, total_loss.item(), fidelity.item(), reg.item()))

        # Live terminal progress
        sys.stdout.write(f"\r[Iteration {i:05d}/{num_iter:05d}] Loss: {total_loss.item():.6f} | PSNR_LR: {psnr_LR:.2f} dB | PSNR_HR: {psnr_HR:.2f} dB")
        sys.stdout.flush()

        if wb_run:
            wb_run.log({
                "iteration": i,
                "loss": total_loss.item(),
                "fidelity_loss": fidelity.item(),
                "reg_loss": reg.item(),
                "psnr_LR": psnr_LR,
                "psnr_HR": psnr_HR,
            })

        # Checkpoints & Previews
        if i % args.log_freq == 0 or i == num_iter - 1:
            preview_img = (out_HR_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(preview_img).save(output_path / f"iter_{i:05d}.png")

            if wb_run:
                wb_run.log({"iteration": i, "reconstructed_HR": wandb.Image(preview_img)})

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

    final = final_metrics(imgs["HR_np"], final_HR_np)
    final_psnr_hr = final["psnr_hr"]
    print(f"[*] Final Super-Resolved Image PSNR: {final_psnr_hr:.4f} dB | SSIM: {final['ssim_hr']:.4f}")

    final_img_arr = (final_HR_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    final_output_file = output_path / f"final_sr_x{args.factor}.png"
    Image.fromarray(final_img_arr).save(final_output_file)
    print(f"[*] Saved final high-resolution image to: {final_output_file}")

    # Save PSNR curve to CSV (first four columns keep the original schema)
    metrics_csv = output_path / "psnr_history.csv"
    np.savetxt(metrics_csv, np.array(psnr_history), delimiter=",",
               header="iteration,psnr_lr,psnr_hr,loss,fidelity_loss,reg_loss", comments="")
    print(f"[*] Saved training metrics log to: {metrics_csv}")

    hist = np.array(psnr_history)
    summary = {
        "status": "complete",
        **final,
        "last_iter_psnr_hr": float(hist[-1, 2]),  # noisy-input output at the last iteration (Stage 1 "last-recorded")
        "best_iter_psnr_hr": float(hist[:, 2].max()),  # oracle, uses the ground truth; informational only
        "best_iter": int(hist[hist[:, 2].argmax(), 0]),
        "final_loss": float(hist[-1, 3]),
        "final_fidelity_loss": float(hist[-1, 4]),
        "final_reg_loss": float(hist[-1, 5]),
        "psnr_bicubic_baseline": float(psnr_bicubic),
        "psnr_nearest_baseline": float(psnr_nearest),
        "runtime_seconds": round(time.time() - start_time, 1),
    }

    if wb_run:
        def to_u8(arr):
            return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)

        diff = np.clip(np.abs(imgs["HR_np"] - final_HR_np) * 10, 0, 1)
        panel = np.concatenate([to_u8(imgs["bicubic_np"]), final_img_arr, to_u8(imgs["HR_np"]), to_u8(diff)], axis=1)
        wb_run.log({
            "final_reconstruction": wandb.Image(final_img_arr, caption=f"PSNR {final_psnr_hr:.2f} dB"),
            "comparison": wandb.Image(panel, caption="bicubic | DIP SR | ground truth | |diff| x10"),
        })
        for key, value in summary.items():
            wb_run.summary[key] = value
        art = wandb.Artifact(f"result-{wb_run.id}", type="sr-result")
        art.add_file(str(final_output_file))
        art.add_file(str(metrics_csv))
        wb_run.log_artifact(art)

    # metrics.json is written last: its presence marks a locally complete run.
    (output_path / "metrics.json").write_text(json.dumps({**summary, "config": config}, indent=2, default=str))


if __name__ == "__main__":
    main()
