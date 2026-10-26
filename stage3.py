"""Helpers for the Stage 3 regularizer experiment (loss x regularizer matrix on Set5 x4).

The notebook DIP_Stage3_Regularizer_Matrix_WandB.ipynb is a thin driver over this module.
Each configuration is one subprocess call of super_resolution_cli.py, which creates exactly one
W&B run for it. W&B is the source of truth for "is this configuration complete"; local files are
only a fallback when W&B is offline.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
EXPERIMENT = "stage3-matrix"

IMAGES = ["baby", "bird", "butterfly", "head", "woman"]
LOSSES = ["l1", "l1-em", "m-l1"]
REGULARIZERS = ["none", "tv", "nn", "tvssah", "bd"]

# Expected Set5 HR sizes (width, height); used only to warn about a wrong download (e.g. LR images).
SET5_SIZES = {"baby": (512, 512), "bird": (288, 288), "butterfly": (256, 256), "head": (280, 280),
              "woman": (228, 344)}

PROTOCOLS = {
    # Same hyper-parameters as the completed Stage 1 baseline, so Stage 3 'none' rows are comparable to it.
    "adapted": {"lr": 0.01, "reg_noise_std": 0.03, "num_iter": 2000,
                "note": "Adapted protocol (Stage 1 settings: lr 0.01, noise 0.03, 2000 it). Not an exact reproduction."},
    # Values quoted for the paper / used by the official main.py defaults.
    "paper": {"lr": 0.001, "reg_noise_std": 0.01, "num_iter": 5500,
              "note": "Paper-style hyper-parameters (lr 0.001, noise 0.01, 5500 it); LR/data pipeline still differs "
                      "from the official code, so not an exact reproduction."},
}

# Regularizer weights: Table 3 of Abalo-Garcia et al. (2025), Set5 row (lambda_TV 0.5, lambda_NN 6, lambda_SSAH 1,
# lambda_BD 0.3); alpha=2.8, beta=0.35 from Sec. 6.2.2. They were tuned by the authors for THEIR protocol
# (lr 0.001, noise 0.01, 5500 it, bicubic LR), so under the "adapted" protocol they are paper values, not tuned ones.
# Caveat: the paper's formulas are un-normalised sums; we use the official code's normalised versions (mean / omega).
DEFAULT_REG_WEIGHTS = {"none": 0.0, "tv": 0.5, "nn": 6.0, "tvssah": 1.0, "bd": 0.3}
WEIGHT_PROVENANCE = {"none": "-", "tv": "paper Table 3 (Set5)", "nn": "paper Table 3 (Set5)",
                     "tvssah": "paper Table 3 (Set5)", "bd": "paper Table 3 (Set5); alpha=2.8, beta=0.35 (Sec. 6.2.2)"}


def run_key(image, loss, reg, seed, protocol, weight, lr_method="pil-lanczos", kernel="lanczos2"):
    key = f"{image}__{loss}__{reg}-{weight:g}__seed{seed}__{protocol}"
    # Defaults keep the keys of the runs already recorded in W&B unchanged.
    if lr_method != "pil-lanczos":
        key += f"__{lr_method}"
    if kernel != "lanczos2":
        key += f"__{kernel}"
    return key


def build_jobs(images, losses, regs, seeds, protocol, reg_weights, lr_method="pil-lanczos", kernel="lanczos2"):
    jobs = []
    for seed in seeds:
        for reg in regs:
            for image in images:
                for loss in losses:
                    w = float(reg_weights[reg])
                    jobs.append({"image": image, "loss": loss, "reg": reg, "weight": w, "seed": seed,
                                 "protocol": protocol, "lr_method": lr_method, "kernel": kernel,
                                 "key": run_key(image, loss, reg, seed, protocol, w, lr_method, kernel)})
    # Unregularised baselines first: they make every later row interpretable even if the session ends early.
    jobs.sort(key=lambda j: (j["reg"] != "none", j["seed"]))
    return jobs


def find_set5(root):
    """Return {name: path} for the five HR images under root; fail loudly instead of guessing."""
    root = Path(root)
    found = {}
    for name in IMAGES:
        cands = [p for p in root.rglob("*") if p.suffix.lower() in (".png", ".bmp") and
                 p.stem.lower().replace("_gt", "").replace("_hr", "") == name and
                 not any("lr" in part.lower().split("_") or "bicubic" in part.lower() or part.lower() == "lr"
                         for part in p.relative_to(root).parts[:-1])]
        if len(cands) != 1:
            raise FileNotFoundError(f"Expected exactly one HR image for {name!r} under {root}, found {cands}")
        found[name] = cands[0]
    from PIL import Image
    for name, path in found.items():
        size = Image.open(path).size
        if size != SET5_SIZES[name]:
            print(f"[warning] {name}: size {size} != expected Set5 HR size {SET5_SIZES[name]} ({path})")
    return found


class Tracker:
    """Answers 'is this configuration complete?' from W&B (online) or local metrics.json (offline)."""

    def __init__(self, entity, project, out_dir):
        self.out_dir = Path(out_dir)
        self.project = project
        self.offline = os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled")
        self.api = None
        if not self.offline:
            import wandb
            self.api = wandb.Api()
            self.entity = entity or self.api.default_entity
            self.path = f"{self.entity}/{project}"
        else:
            self.entity = entity
            self.path = None

    def run_dir(self, key):
        return self.out_dir / key

    def local_complete(self, key):
        d = self.run_dir(key)
        try:
            m = json.loads((d / "metrics.json").read_text())
        except Exception:
            return False
        return m.get("status") == "complete" and any(d.glob("final_sr_x*.png"))

    def wandb_runs(self, key):
        """All W&B runs recorded for this configuration key (any state)."""
        if self.offline:
            return []
        runs = self._query(key)
        return [{"id": r.id, "name": r.name, "state": r.state, "status": r.summary.get("status"), "url": r.url}
                for r in runs]

    def _query(self, key):
        """Runs for a key; a project that does not exist yet (first launch) simply has none."""
        try:
            return list(self.api.runs(self.path, filters={"config.run_key": key}))
        except ValueError as e:
            if "Could not find project" in str(e):
                return []
            raise

    def is_complete(self, key):
        if self.offline:
            return self.local_complete(key)
        # 'finished' alone is not enough: the CLI sets summary.status=complete only after all outputs exist.
        return any(r["state"] == "finished" and r["status"] == "complete" for r in self.wandb_runs(key))

    def n_attempts(self, key):
        return len(self.wandb_runs(key)) if not self.offline else int(self.run_dir(key).exists())


def cli_command(job, cfg, img_path, attempt, tracker):
    out = tracker.run_dir(job["key"])
    extra = {
        "experiment": EXPERIMENT, "stage": "stage3", "run_key": job["key"], "attempt": attempt,
        "protocol": job["protocol"], "protocol_note": cfg["protocol_note"], "dataset": "Set5",
        "scale": cfg["factor"], "fidelity_loss": job["loss"],
        "observation_operator": ("PIL Lanczos (a=3) downsampling of the HR image" if job["lr_method"] == "pil-lanczos"
                                 else "bicubic F.interpolate (official dip-sisr)"),
        "forward_operator": (f"{job.get('kernel', 'lanczos2')} for l1" if job["loss"] == "l1" else "lanczos2+lanczos3 (l1-em / m-l1)"),
        "reg_weight_provenance": WEIGHT_PROVENANCE[job["reg"]],
    }
    cmd = [sys.executable, "-u", str(REPO_DIR / "super_resolution_cli.py"),
           "--img-path", str(img_path), "--factor", str(cfg["factor"]),
           "--num-iter", str(cfg["num_iter"]), "--lr", str(cfg["lr"]),
           "--reg-noise-std", str(cfg["reg_noise_std"]), "--seed", str(job["seed"]),
           "--loss", job["loss"], "--kernel-type", job.get("kernel", "lanczos2"), "--lr-method", job["lr_method"],
           "--regularizer", job["reg"], "--reg-weight", str(job["weight"]),
           "--log-freq", str(cfg["log_freq"]), "--device", cfg.get("device", "auto"),
           "--output-dir", str(out), "--wandb", "--wandb-entity", tracker.entity or "",
           "--wandb-project", tracker.project,
           "--wandb-run-name", f"{cfg.get('name_prefix', 's3')}_{job['image']}_{job['loss']}_{job['reg']}_seed{job['seed']}",
           "--wandb-group", f"stage3-{job['protocol']}-{job['lr_method']}" + ("" if job.get("kernel", "lanczos2") == "lanczos2" else f"-{job['kernel']}"), "--wandb-job-type", cfg.get("job_type", EXPERIMENT),
           "--wandb-tags", ",".join(cfg.get("tags", []) + ["stage3", f"loss:{job['loss']}", f"reg:{job['reg']}",
                                                             f"img:{job['image']}", f"seed:{job['seed']}"]),
           "--extra-config-json", json.dumps(extra)]
    return cmd, out


def run_job(job, cfg, img_path, tracker, log_dir):
    """Run one configuration unless already complete. Returns a status record."""
    key = job["key"]
    if tracker.is_complete(key):
        return {"key": key, "status": "skipped_complete"}
    attempt = tracker.n_attempts(key) + 1
    cmd, out = cli_command(job, cfg, img_path, attempt, tracker)
    if out.exists():  # stale partial output of a failed attempt
        for f in out.glob("*"):
            if f.is_file():
                f.unlink()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{key}__attempt{attempt}.log"
    t0 = time.time()
    with open(log_path, "w", buffering=1) as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_DIR).returncode
    elapsed = time.time() - t0
    ok = rc == 0 and tracker.local_complete(key)
    rec = {"key": key, "status": "complete" if ok else "failed", "returncode": rc, "attempt": attempt,
           "elapsed_s": round(elapsed, 1), "log": str(log_path)}
    if not ok:
        tail = Path(log_path).read_text()[-1500:]
        rec["error_tail"] = tail
    return rec


def collect_results(jobs, tracker):
    """One row per planned configuration: best completed run, or the reason it is missing/failed."""
    rows = []
    for job in jobs:
        row = {k: job[k] for k in ("key", "image", "loss", "reg", "weight", "seed", "protocol", "lr_method", "kernel")}
        if tracker.offline:
            if tracker.local_complete(job["key"]):
                m = json.loads((tracker.run_dir(job["key"]) / "metrics.json").read_text())
                row.update({k: m.get(k) for k in _METRICS}); row["status"] = "complete"
            else:
                row["status"] = "missing_or_failed"
        else:
            runs = tracker._query(job["key"])
            done = [r for r in runs if r.state == "finished" and r.summary.get("status") == "complete"]
            if done:
                r = max(done, key=lambda r: r.created_at)
                row.update({k: r.summary.get(k) for k in _METRICS})
                row.update({"status": "complete", "wandb_run": r.id, "url": r.url})
            elif runs:
                r = max(runs, key=lambda r: r.created_at)
                row.update({"status": f"failed:{r.state}", "wandb_run": r.id, "url": r.url,
                            "error": r.summary.get("error")})
            else:
                row["status"] = "missing"
        rows.append(row)
    return pd.DataFrame(rows)


_METRICS = ["psnr_hr", "ssim_hr", "psnr_y_border4", "last_iter_psnr_hr", "best_iter_psnr_hr", "best_iter",
            "final_loss", "final_fidelity_loss", "final_reg_loss", "runtime_seconds"]


def add_deltas(df, metric="psnr_hr"):
    """Delta of each row vs the unregularised run with the same image/loss/seed."""
    ok = df[df.status == "complete"]
    base = ok[ok.reg == "none"].set_index(["image", "loss", "seed"])[metric]
    out = df.copy()
    out[f"delta_{metric}"] = [
        (r[metric] - base.get((r.image, r.loss, r.seed), np.nan)) if r.status == "complete" else np.nan
        for _, r in out.iterrows()]
    return out


def summarize(df, metric="psnr_hr"):
    """Answers the four Stage 3 questions from a complete/partial results frame."""
    d = add_deltas(df, metric)
    col = f"delta_{metric}"
    reg = d[(d.status == "complete") & (d.reg != "none")]
    mean_delta = reg.pivot_table(index="loss", columns="reg", values=col, aggfunc="mean")
    wins = reg.assign(win=reg[col] > 0).pivot_table(index="loss", columns="reg", values="win", aggfunc="sum")
    count = reg.pivot_table(index="loss", columns="reg", values=col, aggfunc="count")
    per_image = reg.pivot_table(index=["loss", "reg"], columns="image", values=col, aggfunc="mean")
    baseline = d[(d.status == "complete") & (d.reg == "none")].pivot_table(
        index="loss", columns="image", values=metric, aggfunc="mean")
    return {"mean_delta": mean_delta, "images_improved": wins, "n_complete": count, "per_image_delta": per_image,
            "baseline": baseline, "frame": d}
