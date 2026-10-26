"""Regularizers for DIP super-resolution, ported from the paper's official code.

Source: https://github.com/capo-urjc/dip-sisr (commit e2c80f0, MIT License,
Copyright (c) 2024 CAPO Research Group at URJC), files ``losses/functions.py`` and
``utils/dip_utils.py``. The function bodies are copied unchanged so that the
regularizer values match the official implementation; tests/test_regularizers.py
checks this numerically against the official file.

Names used by this repo's CLI -> official names:
    none    -> (no term)
    tv      -> p_tv_norm_isotropic(fadf="forward", p=p_tv)      weight: tv_param
    nn      -> schatten_norm(p=p_nn)  (p=1: nuclear norm)        weight: nn_param
    tvssah  -> SSAHTV_norm(mu=0.01, fadf="central")              weight: ssahtv_param
    bd      -> schiavi_norm(alpha, beta)                         weight: schiavi_param
"""
import torch
from torch import Tensor

REGULARIZERS = ("none", "tv", "nn", "tvssah", "bd")

# Defaults taken from the argparse defaults of the official main.py.
OFFICIAL_DEFAULTS = {"p_tv": 1.0, "p_nn": 1.0, "ssah_mu": 0.01, "bd_alpha": 2.8, "bd_beta": 0.35,
                     "ssahtv_weight": 0.8}


def calculate_gradients(tensor: Tensor, fadf: str) -> tuple:
    assert fadf in ["forward", "central", "backward"], "FADF not valid"

    delta_x = torch.zeros_like(tensor)
    delta_y = torch.zeros_like(tensor)

    if fadf == "central":

        delta_x[..., :, 1: -1] = 0.5 * tensor[..., :, 2:] - 0.5 * tensor[..., :, :-2]
        delta_y[..., 1: -1, :] = 0.5 * tensor[..., 2:, :] - 0.5 * tensor[..., :-2, :]

    elif fadf == "forward":

        delta_x[..., :, :-1] = tensor[..., :, 1:] - tensor[..., :, :-1]
        delta_y[..., :-1, :] = tensor[..., 1:, :] - tensor[..., :-1, :]

    elif fadf == "backward":

        delta_x[..., :, 1:] = tensor[..., :, 1:] - tensor[..., :, :-1]
        delta_y[..., 1:, :] = tensor[..., 1:, :] - tensor[..., :-1, :]

    return delta_x, delta_y


def p_tv_norm_isotropic(tensor: Tensor, p: float, fadf: str) -> Tensor:
    assert fadf in ["forward", "central", "backward"], "FADF not valid"
    assert p > 0, "p value must be greater than 0"

    delta_x, delta_y = calculate_gradients(tensor, fadf)

    mod_p = torch.pow(delta_x**2 + delta_y**2 + 1e-6, p/2)
    tv_loss = mod_p.mean()

    return tv_loss


def schatten_norm(tensor: Tensor, p: float) -> Tensor:
    """
    With p=1 we have nuclear norm
    """
    assert p > 0, "p value must be greater than 0"

    b, c, w, h = tensor.size()
    omega = c * w * h

    nuclear_norms = torch.norm(tensor, p='nuc', dim=(2, 3))
    nuclear_loss = torch.norm(nuclear_norms, p=p, dim=1) / omega

    return nuclear_loss


def SSAHTV_norm(tensor: Tensor, mu: float, fadf: str):
    assert fadf in ["forward", "central", "backward"], "FADF not valid"

    delta_x, delta_y = calculate_gradients(tensor, fadf=fadf)

    b, c, h, w = tensor.size()

    squared_mod = delta_x**2 + delta_y**2
    G = torch.sqrt(squared_mod.sum(1, keepdims=False) + 1e-6)

    tau = (1 / (1 + mu * G))  # (14)

    W = tau  # (15) ??

    ssah_tv = torch.sum(W * G)  # (16)
    ssah_tv = ssah_tv/(c * h * w)

    return ssah_tv


def schiavi_norm(tensor, alpha: float, beta: float, p: float = 1):

    delta_x, delta_y = calculate_gradients(tensor, fadf="central")

    b, c, h, w = tensor.size()

    squared_mod = delta_x**2 + delta_y**2
    mod = torch.sqrt(squared_mod.sum(1, keepdims=False) + 1e-6)

    schiavi_term = (alpha * torch.exp(-mod) - beta)  # (14)

    schiavi = torch.sum(schiavi_term * mod)  # (16)

    schiavi = schiavi / (c * h * w)

    return schiavi


def regularization_term(name: str, hr: Tensor, weight: float, p_tv: float = OFFICIAL_DEFAULTS["p_tv"],
                        p_nn: float = OFFICIAL_DEFAULTS["p_nn"], ssah_mu: float = OFFICIAL_DEFAULTS["ssah_mu"],
                        bd_alpha: float = OFFICIAL_DEFAULTS["bd_alpha"],
                        bd_beta: float = OFFICIAL_DEFAULTS["bd_beta"]) -> Tensor:
    """Weighted regularizer added to the fidelity loss; mirrors utils/dip_utils.loss_fn of the official code."""
    if name == "none" or weight == 0:
        return hr.new_zeros(())
    if name == "tv":
        return weight * p_tv_norm_isotropic(hr, p=p_tv, fadf="forward")
    if name == "nn":
        return weight * schatten_norm(hr, p=p_nn)
    if name == "tvssah":
        return weight * SSAHTV_norm(hr, mu=ssah_mu, fadf="central")
    if name == "bd":
        return weight * schiavi_norm(hr, alpha=bd_alpha, beta=bd_beta)
    raise ValueError(f"Unknown regularizer {name!r}; expected one of {REGULARIZERS}")
