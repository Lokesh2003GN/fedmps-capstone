"""Adapting the end of the frozen backbone instead of its output.

The MLP adapter (adapter.py) only ever sees the 512-d embedding of a degraded
face, after the backbone has already compressed away what the camera blurred.
This module federates something one step earlier: the last two layers of
w600k_r50, which turn the 512 x 7 x 7 feature map of the last residual stage
into the embedding.

    layer-4 feature map [512, 7, 7]   (saved per image by 01b_prepare_stream.py)
        -> bn2: per-channel scale and shift     (trained: 1,024 parameters)
        -> fc: 25,088 -> 512                    (frozen, plus a trained
                                                 rank-r update B @ A)
        -> features BatchNorm                   (frozen)
        -> L2 normalise

Both layers are linear, so with the update at zero this reproduces the
backbone's own embedding exactly (01b checks this to 1e-5): round 0 is the
frozen backbone, as it is for the MLP adapter. B starts at zero, as in LoRA,
and the bn2 scale and shift start at the backbone's values.

Only the trained tensors are parameters. The frozen ones are plain attributes,
so state_dict(), and with it every federated update, DP clip and secure-
aggregation mask, covers only what is trained.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AdapterConfig

FEAT_CHANNELS, FEAT_HW = 512, 49
FEAT_DIM = FEAT_CHANNELS * FEAT_HW
BN_EPS = 1e-5

_FROZEN: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}


def _onnx_path() -> str:
    from insightface.utils import storage

    from ..config import BackboneConfig

    bc = BackboneConfig()
    return os.path.join(str(storage.ensure_available("models", bc.insightface_model)),
                        bc.insightface_onnx)


def frozen_head(device: str) -> Dict[str, torch.Tensor]:
    """The backbone's own bn2, fc and features-BN tensors, loaded once per device."""
    path = _onnx_path()
    key = (path, str(device))
    if key not in _FROZEN:
        import onnx
        from onnx import numpy_helper

        init = {t.name: numpy_helper.to_array(t)
                for t in onnx.load(path).graph.initializer}
        dev = torch.device(device)

        def t(name: str) -> torch.Tensor:
            return torch.from_numpy(np.array(init[name], dtype=np.float32)).to(dev)

        feat_scale = t("features.weight") / torch.sqrt(t("features.running_var") + BN_EPS)
        _FROZEN[key] = {
            "bn2_gamma": t("bn2.weight"),
            "bn2_beta": t("bn2.bias"),
            "bn2_mean": t("bn2.running_mean"),
            "bn2_inv_std": 1.0 / torch.sqrt(t("bn2.running_var") + BN_EPS),
            "fc_w_t": t("fc.weight").t().contiguous(),          # [25088, 512]
            "fc_b": t("fc.bias"),
            "feat_scale": feat_scale,
            "feat_shift": t("features.bias") - t("features.running_mean") * feat_scale,
        }
    return _FROZEN[key]


class HeadAdapter(nn.Module):
    """bn2 scale/shift and a low-rank fc update, on layer-4 features."""

    kind = "head"

    def __init__(self, cfg: AdapterConfig, device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        frozen = frozen_head(device)
        self._frozen = frozen
        r = int(cfg.lora_rank)
        self.bn_gamma = nn.Parameter(frozen["bn2_gamma"].clone())
        self.bn_beta = nn.Parameter(frozen["bn2_beta"].clone())
        self.lora_a = nn.Parameter(torch.empty(r, FEAT_DIM, device=frozen["fc_b"].device))
        self.lora_b = nn.Parameter(torch.zeros(frozen["fc_b"].shape[0], r,
                                               device=frozen["fc_b"].device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = float(cfg.lora_alpha) / max(r, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is [N, 25088] layer-4 features; returns [N, 512] unit vectors."""
        f = self._frozen
        s = self.bn_gamma * f["bn2_inv_std"]
        shift = self.bn_beta - f["bn2_mean"] * s
        x = x.reshape(-1, FEAT_CHANNELS, FEAT_HW) * s[None, :, None] + shift[None, :, None]
        x = x.reshape(x.shape[0], FEAT_DIM)
        z = x @ f["fc_w_t"] + f["fc_b"] + (x @ self.lora_a.t()) @ self.lora_b.t() * self.scale
        z = z * f["feat_scale"] + f["feat_shift"]
        return F.normalize(z, p=2, dim=-1)

    @torch.no_grad()
    def embed_rows(self, feat: np.ndarray, rows: np.ndarray,
                   batch_size: int = 8192,
                   cache: Optional[torch.Tensor] = None) -> np.ndarray:
        """Embed feature rows `rows` (sorted) of a memory-mapped [N, 25088] array.

        `cache`, if given, is those rows already on the device in float16,
        which turns a per-round evaluation into a few matrix products.
        """
        self.eval()
        device = self.lora_b.device
        out = np.empty((rows.size, 512), dtype=np.float32)
        for i in range(0, rows.size, batch_size):
            if cache is not None:
                x = cache[i:i + batch_size].float()
            else:
                x = torch.from_numpy(np.asarray(feat[rows[i:i + batch_size]])).to(device).float()
            out[i:i + batch_size] = self(x).cpu().numpy()
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def payload_bytes(self) -> int:
        """Size of one federated update, as float32 on the wire."""
        return self.n_params() * 4


def build_head(cfg: AdapterConfig, device: str = "cpu") -> HeadAdapter:
    return HeadAdapter(cfg, device)
