# Загрузка весов модели и выполнение инференса
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from model.unet3d import UNet3D
from services import preprocessing

log = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights: str | Path, device: str | torch.device = "cpu", base: int = 16) -> UNet3D:
    # Загрузить обученную 3D U-Net
    device = torch.device(device)
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    net = UNet3D(base=base)
    net.load_state_dict(state)
    net.to(device).eval()
    log.info("Модель загружена: %s", weights)
    return net


def predict_mask(net: UNet3D, hu_zyx: np.ndarray, device: torch.device,
                 target_shape=preprocessing.TARGET_SHAPE,
                 window=preprocessing.WINDOW) -> np.ndarray:
    # Предсказать маску печени (Z, Y, X) в исходном разрешении
    xyz = np.transpose(hu_zyx, (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
    vol = preprocessing.prepare_for_model(xyz, target_shape, window)
    x = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(net(x))[0, 0].cpu().numpy()

    mask_small = (prob > 0.5).astype(np.uint8)
    if tuple(mask_small.shape) != tuple(xyz.shape):
        mask_xyz = preprocessing.resize_volume(mask_small, xyz.shape, order=0)
    else:
        mask_xyz = mask_small
    mask = np.transpose(mask_xyz, (2, 1, 0))  # (X, Y, Z) -> (Z, Y, X)
    return (mask > 0.5).astype(np.uint8)
