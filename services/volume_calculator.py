# Расчёт объёма печени и 3D-постобработка маски
from __future__ import annotations

import numpy as np
from scipy import ndimage


def voxel_volume_mm3(affine: np.ndarray) -> float:
    # Объём одного вокселя (мм³)
    return float(abs(np.linalg.det(affine[:3, :3])))


def compute_volume(mask: np.ndarray, affine: np.ndarray) -> float:
    # Объём печени в миллилитрах
    return float(mask.astype(np.uint32).sum() * voxel_volume_mm3(affine) / 1000.0)


def postprocess_mask(mask: np.ndarray, min_voxels: int = 500,
                     closing_iterations: int = 1) -> np.ndarray:
    # 3D-морфология
    out = mask.astype(bool)
    if not out.any():
        return out.astype(np.uint8)

    # крупнейший связный объект (печень — единый орган)
    labels, n = ndimage.label(out)
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        out = labels == int(np.argmax(sizes))

    # заполнить внутренние полости
    out = ndimage.binary_fill_holes(out)

    # морфологическое замыкание
    if closing_iterations > 0:
        out = ndimage.binary_closing(out, iterations=closing_iterations)

    # удалить мелкие объекты
    if min_voxels > 1:
        labels, n = ndimage.label(out)
        if n > 0:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= min_voxels
            keep[0] = False
            out = keep[labels]

    return out.astype(np.uint8)


def slice_areas_mm2(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    # Площадь печени на каждом срезе (мм²)
    sy = float(abs(affine[1, 1]))
    sx = float(abs(affine[0, 0]))
    return mask.astype(np.uint32).sum(axis=(1, 2)) * sy * sx


def slice_volume_ml(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    # Вклад каждого среза в объём (мл)
    return slice_areas_mm2(mask, affine) * float(abs(affine[2, 2])) / 1000.0
