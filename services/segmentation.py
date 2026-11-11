# Обучение нейросети (загрузка весов и предсказание маски печени)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from services import preprocessing

log = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights: str | Path, device: str | torch.device = "cpu", base: int = 16):
    # Загрузить обученную 3D U-Net
    from model.unet3d import UNet3D

    device = torch.device(device)
    state = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    net = UNet3D(base=base)
    net.load_state_dict(state, strict=False)
    net.to(device).eval()
    log.info("Модель загружена: %s", weights)
    return net


def predict_mask(net, hu_zyx: np.ndarray, device: torch.device,
                 target_shape=preprocessing.TARGET_SHAPE,
                 window=preprocessing.WINDOW):
    # Предсказать маску печени. Возвращает (mask, prob)
    xyz = np.transpose(hu_zyx, (2, 1, 0))  # (Z, Y, X) -> (X, Y, Z)
    vol = preprocessing.prepare_for_model(xyz, target_shape, window)
    x = torch.from_numpy(np.ascontiguousarray(vol)).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = net(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (prob > 0.5).astype(np.uint8)
    if tuple(mask_small.shape) != tuple(xyz.shape):
        mask_xyz = preprocessing.resize_volume(mask_small, xyz.shape, order=0)
        prob_xyz = preprocessing.resize_volume(prob, xyz.shape, order=1)
    else:
        mask_xyz = mask_small
        prob_xyz = prob
    mask = np.transpose(mask_xyz, (2, 1, 0))  # (X, Y, Z) -> (Z, Y, X)
    prob = np.transpose(prob_xyz, (2, 1, 0))  # (X, Y, Z) -> (Z, Y, X)
    return (mask > 0.5).astype(np.uint8), prob


def segment(net, hu_zyx: np.ndarray, affine: np.ndarray, device: torch.device,
            progress=None) -> "SegmentationResult":
    # Предсказание + постобработка + объём + уверенность
    from services.volume_calculator import compute_volume, postprocess_mask

    if progress:
        progress(25)
    mask, prob = predict_mask(net, hu_zyx, device)
    if progress:
        progress(75)

    mask = postprocess_mask(mask)
    if progress:
        progress(90)

    volume_ml = compute_volume(mask, affine)

    # Уверенность
    if mask.sum() > 0:
        confidence = float(prob[mask > 0].mean()) * 100.0
    else:
        confidence = 0.0
    if progress:
        progress(100)

    return SegmentationResult(
        mask=mask, volume_ml=volume_ml, confidence=confidence,
    )


class SegmentationResult:
    # Результат сегментации
    def __init__(self, mask: np.ndarray, volume_ml: float, confidence: float = 0.0) -> None:
        self.mask = mask
        self.volume_ml = volume_ml
        self.voxel_count = int(mask.sum())
        self.confidence = confidence
