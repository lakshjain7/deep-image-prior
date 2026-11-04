# Handoff: DIP super-resolution research (operators, regularizers, baseline reproduction)

Paste the "Prompt for the new chat" section into a new chat, and attach this file plus the paper PDF.

---

## Prompt for the new chat

> I'm continuing a research project with my professor (Ashesh) on Deep Image Prior (DIP) single-image super-resolution. The full context is in HANDOFF.md (attached) and in my repo https://github.com/lakshjain7/deep-image-prior, branch `claude/epic-ride-rnqbk4`. The paper we build on is Abalo-García et al. (2025), "Unsupervised Deep Image Prior-Based Neural Networks for Single Image Super-Resolution", Expert Systems, DOI 10.1111/exsy.70142 (PDF attached); its code is https://github.com/capo-urjc/dip-sisr (commit e2c80f0).
>
> Read HANDOFF.md first, then the repo files it lists. Do not repeat finished experiments. All runs are tracked in W&B (entity `deepimageprior`, project `dip-sr-regularizers`); I run everything on free Google Colab GPUs.
>
> My current task, from the last meeting: **reproduce the DIP baseline exactly on one Set5 image (baby, ×4) before re-running any ablation**, using `DIP_Baseline_Reproduction.ipynb`. Then re-run the downsampler and regularizer ablations from that baseline, add visual comparisons, and prepare a clear step-by-step explanation of aliasing inside the DIP loss pipeline.
>
> Be concise, give me exact cells to run and where, explain results in plain terms with numbers, and separate verified facts from open questions.

---

## 1. People, goal, status

- **Professor's original email:** understand the paper's regularizers and downsampling operators, swap the downsampling operator in my DIP setup, and check whether performance differs. Building on the method only makes sense if it does.
- **Last meeting (after we presented):** results trail the paper's numbers, so they can't be interpreted until the baseline is reproduced exactly.
  - **Priority now:** reproduce DIP on one image (baby), commit that state, then re-run the ablations.
  - **Postponed:** the learned / weighted-regularizer idea.
  - **Professor's points:**
    - Set5 has only HR images, and the LR is made with a downsampler. This is correct, and it's what we do.
    - The authors' GitHub code should show the exact setup.
    - He found the aliasing explanation unclear and wants step-by-step reasoning inside the DIP loss.
    - Visual inspection should show whether low PSNR comes from artifacts or from blur.
- **Next steps from the meeting:**
  - [ ] (me) Confirm DIP creates Set5 LR by downsampling HR. Yes: the original DIP code uses PIL Lanczos in `utils/sr_utils.py::load_LR_HR_imgs_sr`, and the authors' code uses `F.interpolate` bicubic in `main.py`.
  - [ ] (me) Reproduce the exact DIP PSNR on one Set5 image using their code, then commit. Notebook: `DIP_Baseline_Reproduction.ipynb`.
  - [ ] (me) Re-run the downsampler and regularizer ablations from the reproduced baseline.
  - [ ] (me) Read about aliasing and antialiasing in SR. Section 6 below has the explanation.
  - [ ] (me) Add visualizations of predictions across configurations. Part D of the reproduction notebook does this.
  - [ ] (Ashesh) Look for more recent SISR methods.

## 2. Code and where things are

- **Repo:** `lakshjain7/deep-image-prior`, branch `claude/epic-ride-rnqbk4`. It's a fork of Ulyanov's original DIP code.
- **Files:**
  - `super_resolution_cli.py` runs one experiment. Main flags:
    - `--loss mse|l1|l1-em|m-l1`, `--kernel-type lanczos2|lanczos3`
    - `--lr-method pil-lanczos|bicubic|bicubic-aa`, `--regularizer none|tv|nn|tvssah|bd`, `--reg-weight`
    - W&B logging covers the config, per-iteration curves, final image, comparison panel and an artifact `result-<run id>`.
    - Metrics: `psnr_hr` (RGB, data_range 1), `ssim_hr`, `psnr_y_border4` (Y channel with a 4-px border, the DIP paper's protocol), `psnr_LR` (fit to the input), `psnr_bicubic_baseline`.
    - A run counts as complete only once `metrics.json` is written and W&B has `summary.status = complete`.
  - `utils/regularizers.py` holds TV, NN, TVSSAH and BD, copied unchanged from the authors' `losses/functions.py`. `tests/test_regularizers.py` checks they match bit-for-bit.
  - `utils/sr_utils.py::load_LR_HR_imgs_sr(..., lr_method)` makes the LR image:
    - `pil-lanczos` is the default and the original DIP behaviour.
    - `bicubic` is `F.interpolate(..., mode="bicubic", align_corners=False)` without antialiasing, as in the authors' code.
    - `bicubic-aa` is the same call with `antialias=True`.
  - `stage3.py` builds jobs (`build_jobs`), handles resume (`Tracker.is_complete` via the W&B `config.run_key`), runs jobs (`run_job`) and collects results.
    - Run key format: `<image>__<loss>__<reg>-<weight>__seed<seed>__<protocol>[__<lr_method>][__<kernel>]`.
    - Protocols:
      - `adapted`: lr 0.01, noise 0.03, 2000 iterations. These are the original DIP defaults.
      - `paper`: lr 0.001, noise 0.01, 5500 iterations.
    - Weights (paper Table 3, Set5): TV 0.5, NN 6, TVSSAH 1, BD 0.3; α 2.8, β 0.35.
  - `evaluate_set5.py` does the Y-channel PSNR with a border crop. Its `PAPER_PSNR` holds Ulyanov's ×4 numbers: baby 31.49, bird 31.80, butterfly 26.23, head 31.04, woman 28.93.
- **Notebooks** (open them from GitHub in Colab; section 1 always pulls the latest code):
  - `DIP_Stage3_Regularizer_Matrix_WandB.ipynb` is the runner.
    - Run sections 1, 2, 3 and 4, skip 5 and 6, then run 7.
    - Set `EXPERIMENT` in section 3: `matrix`, `operator`, `paper-test`, `mismatch-A`, `mismatch-B`, `mismatch`, `paper-clean`, `paper-clean-rest`. All of these are done except `paper-clean-rest`.
    - Never run the same `EXPERIMENT` in two notebooks at once.
  - `DIP_Results_Summary.ipynb` uses no GPU and builds every results table from W&B. Sections:
    - A: losses
    - B: regularizers vs. the paper
    - C: operators
    - D: why aliased input fails
    - E: the aliasing study
    - F: clean input under the paper's settings
  - **`DIP_Baseline_Reproduction.ipynb`** is the current task and hasn't been run yet:
    - Part A: the authors' code, patched minimally, on baby. L1, no regularizer, paper settings. Target: Table A5 baby **33.35**.
    - Part B: our code with the original DIP setup (MSE, Lanczos-2, PIL-Lanczos LR, 2000 iterations, lr 0.01, noise 0.03). Target: Ulyanov baby **31.49** on the Y channel.
    - Part C: our code with the authors' pipeline. It should match Part A.
    - Part D: visual comparison grid with zoomed crops.
    - Part E: summary vs. both papers.
- **Meeting doc:** "DIP Super-Resolution: Operators and Regularizers — Meeting Brief", https://claude.ai/artifact/U6sVd6wmUzNCMZ4W2UkC2S.
- **W&B:** entity `deepimageprior`. Projects:
  - `dip-sr-regularizers` holds every Stage 3+ run.
  - `deep-image-prior` holds the original Stage 1 runs.
  - The API key is in the Colab secret `WANDB_API_KEY`.
- **Colab constraints:**
  - The free GPU lasts about 2 to 3 hours per session. Resume works across sessions and accounts as long as the same W&B key is used.
  - Run times at 2000 iterations: baby about 10 min, other images about 3 min. At 5500 iterations: about 7 to 8 min, and NN about 10 to 13 min.

## 3. Problems found (and status)

1. **The authors' `main.py` cannot run unchanged.** The reproduction notebook patches all four of these and prints each patch.
   - `datasets/` has no `__init__.py`, so Colab's Hugging Face `datasets` package hides it and the import fails with `ModuleNotFoundError: No module named 'datasets.NaturalColor'`. Verified locally.
   - `CUDA_VISIBLE_DEVICES = "3"` is hard-coded.
   - `--factor` is parsed as a float, so `get_kernel` crashes: `'float' object cannot be interpreted as an integer`. Verified with the identical downsampler.
   - The output image is never saved.
   - `--factor` defaults to 2, the default `ssahtv_param` is 0.8 (TVSSAH on), and the default `method` is 1 (L1-EM).
2. **The Stage 1 metric was wrong.** It averaged `psnr_lr` (the fit to the LR input, about 40 dB) instead of `psnr_hr`. Fixed. Corrected values: L1 28.36, L1-EM 28.56, M-L1 28.36.
3. **Our baseline isn't the original DIP.** We used L1, while the original DIP uses MSE. Our LR is made with PIL Lanczos, while the authors' code uses non-antialiased bicubic. Training, cropping (we crop to multiples of 32) and the metric also differ. The exact 2025 setup (scale factor, preprocessing) is not in the paper.
4. **Two different "DIP baselines" for baby:** Ulyanov 2017 reports 31.49 (Y channel, border crop), and the 2025 paper's Table A5 reports 33.35. Our RGB value was 30.83. Our `psnr_y_border4` for that run is in W&B but hasn't been checked yet.
5. **Regularizer behaviour in the authors' code** (computed from their constants on [0, 1] images):
   - TVSSAH's edge weight 1/(1 + 0.01·G) stays between 0.992 and 1.0, so it behaves almost exactly like channel-coupled TV.
   - BD's weight 2.8·e^(−G) − 0.35 only turns negative for G > ln 8 ≈ 2.08, and G ≤ 1.22, so it never sharpens. It's just a weak smoother.

## 4. Results so far (Set5 ×4, one seed unless stated; PSNR vs. the real image, in dB)

**Losses** (2 seeds, L1, no regularizer, PIL-Lanczos LR, adapted settings): L1-EM beats L1 on 5 of 5 images, by +0.02 to +0.43 (mean 28.56 vs. 28.36). M-L1 is no better (28.36). The paper reports L1-EM +0.41 and M-L1 +0.23.

**75-run regularizer matrix** (PIL-Lanczos LR, adapted settings). Gain over the same loss without a regularizer:

| Loss | BD | NN | TV | TVSSAH |
|---|---|---|---|---|
| L1 | −0.54 | −0.86 | −0.08 | −0.11 |
| L1-EM | −0.58 | −0.95 | −0.15 | −0.17 |
| M-L1 | −0.37 | −0.82 | −0.00 | −0.04 |

The paper (Table 4) reports gains over L1 of +0.79 (BD), +1.15 (NN), +1.41 (TV) and +1.47 (TVSSAH) for the L1 loss. Our L1 baselines are baby 30.83, bird 30.13, butterfly 25.07, head 28.90 and woman 27.45, with a mean of 28.48. The paper's Table A5 has 33.35, 33.48, 24.79, 29.74 and 29.24, with a mean of 30.12.

**Operator experiment** (L1, no regularizer):

| LR made with | Model kernel | baby | bird | butterfly | head | woman |
|---|---|---|---|---|---|---|
| PIL Lanczos | Lanczos-2 | 30.83 | 30.13 | 25.07 | 28.90 | 27.45 |
| PIL Lanczos | Lanczos-3 | 31.23 | 30.51 | 25.56 | 29.08 | 27.68 |
| bicubic, no antialiasing | Lanczos-2 | 26.97 | 24.35 | 16.40 | 24.47 | 21.16 |
| bicubic, no antialiasing | Lanczos-3 | 27.19 | 25.28 | 16.75 | 25.31 | 21.87 |

The way the LR is made changes PSNR by 3.9 to 8.7 dB. Lanczos-3 beats Lanczos-2 in all 10 cells, by +0.18 to +0.93 dB.

**Aliasing study** (L1, adapted settings):

| LR input | Antialiased? | Mean, no regularizer | Mean TV gain | TV improved | DIP minus bicubic upsampling |
|---|---|---|---|---|---|
| PIL Lanczos | yes | 28.48 | −0.08 | 3 of 5 | +1.54 |
| bicubic-aa | yes | 28.69 | −0.34 | 0 of 5 | +2.03 |
| bicubic | no | 22.67 | +1.84 | 5 of 5 | −3.34 |

On aliased bicubic input, over 4 images (bird, butterfly, head, woman), the regularizers gain TVSSAH +2.00, TV +1.88, NN +1.83 and BD +0.69. That's the same order as the paper. On PIL Lanczos input, the same 4 images give TVSSAH −0.19, TV −0.15, NN −1.11 and BD −0.67.

**Paper training settings** (5500 iterations, lr 0.001, noise 0.01):
- **Clean PIL-Lanczos input** (4 images): the no-regularizer means are 28.15 none, 27.65 TV and 26.79 NN, so TV −0.50 and NN −1.36. The baseline is +0.16 to +0.36 above the adapted settings, and the fit to the input is about 50 dB.
- **Aliased input:** butterfly no regularizer 15.94, TV 17.58; head no regularizer 24.75, TV 26.00.
- TVSSAH and BD under the paper's settings on clean input (`paper-clean-rest`) haven't been run.

**Fit to input vs. real image:**
- Adapted settings: fit is 36 to 42 dB for both clean and aliased input.
- Paper settings: fit is about 50 dB for both.
- A high fit hurts only when the input is aliased. DIP on aliased input is 3 to 4 dB below plain bicubic upsampling.

## 5. Interpretation (what we claim, what we don't)

- **Claim (in our setup):**
  - The downsampling operator matters, by 4 to 9 dB.
  - The paper's regularizers help only when the LR image is aliased. With antialiased inputs they give no gain, under both training settings.
  - The reason: DIP's Lanczos shrinking can't produce aliasing, so matching an aliased input forces spurious detail into the output. The regularizers suppress that detail.
- **Don't claim:**
  - That this explains the paper's own numbers. Their butterfly baseline (24.79) looks like our clean result (25.07), not our aliased one (16.40), and their pipeline isn't reproduced yet.
  - Anything from gaps under about 0.3 dB, since most results are one seed.

## 6. Aliasing, step by step in the DIP loss (for the professor)

1. **HR image x** comes from Set5.
2. **The LR input** is y = D_obs(x), shrinking 4×. Antialiased means blur first, then subsample. Aliased means subsample without blurring, so detail finer than the new pixel size folds into fake coarse patterns. **This is the only step where aliasing can happen.**
3. **DIP outputs** x̂ = f_θ(z).
4. **The loss** is ‖D_model(x̂) − y‖, where D_model is Lanczos and always antialiased.
5. **If y is antialiased:** an x̂ close to x satisfies the loss.
6. **If y is aliased:** no natural x̂ satisfies it, because D_model can't create aliasing. DIP adds spurious high-frequency content to x̂ whose blurred-and-shrunk version mimics the aliasing, so x̂ moves away from x and PSNR drops below plain upsampling.
7. **The regularizers** penalize that spurious content, which is why they help only in case 6.

## 7. To do, in order

1. Run `DIP_Baseline_Reproduction.ipynb` on a GPU: Setup, then A, B, C (about 70 min), then D and E.
   - **A ≈ 33.35:** the authors' pipeline is reproduced.
   - **A far below 33.35:** try `FACTOR = 2`, or ask the authors.
   - **B's `psnr_y_border4` ≈ 31.49:** the original DIP is reproduced.
   - **C vs. A:** checks that our copy of their pipeline is faithful.
   - Part A has never been executed. If it fails, read the log tail it prints.
2. Commit the reproduced baseline state. Record the settings and numbers.
3. Re-run the operator and regularizer ablations from the reproduced baseline: the same presets with the reproduced settings, plus visual grids (Part D) per configuration.
4. Prepare the aliasing explanation (section 6) with a picture: an LR made with and without antialiasing, side by side.
5. Later, once the baseline is reproduced: the learned weighting of operators or regularizers, and matching or learning the forward operator. Prediction: DIP recovers on aliased input without any regularizer.
6. Optional: a second seed, Set14, and `paper-clean-rest` (TVSSAH and BD on clean input, paper settings, about 70 min).

## 8. Communication preferences

- Be concise and give exact cells and where to run them.
- Don't hedge everything, but say clearly what's verified vs. open.
- Plain explanations with numbers; no jargon dumps.
- Don't tell me to re-run finished work.
