"""Checks that utils/regularizers.py reproduces the official dip-sisr implementation.

Set DIP_SISR_DIR to a checkout of https://github.com/capo-urjc/dip-sisr (the test is skipped otherwise).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import regularizers as mine  # noqa: E402

OFFICIAL = Path(os.environ.get("DIP_SISR_DIR", "/home/user/capo-urjc/dip-sisr")) / "losses" / "functions.py"


@pytest.mark.skipif(not OFFICIAL.exists(), reason="official dip-sisr checkout not found")
def test_matches_official():
    spec = importlib.util.spec_from_file_location("official_functions", OFFICIAL)
    off = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(off)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 96)
    assert torch.equal(mine.p_tv_norm_isotropic(x, 1.0, "forward"), off.p_tv_norm_isotropic(x, 1.0, "forward"))
    assert torch.equal(mine.schatten_norm(x, 1.0), off.schatten_norm(x, 1.0))
    assert torch.equal(mine.SSAHTV_norm(x, 0.01, "central"), off.SSAHTV_norm(x, 0.01, "central"))
    assert torch.equal(mine.schiavi_norm(x, 2.8, 0.35), off.schiavi_norm(x, 2.8, 0.35))


def test_dispatch_and_gradients():
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    for name in mine.REGULARIZERS:
        t = mine.regularization_term(name, x, 0.5)
        if name == "none":
            assert t.item() == 0
        else:
            t.backward()
            assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
            x.grad = None
    with pytest.raises(ValueError):
        mine.regularization_term("bogus", x, 1.0)
