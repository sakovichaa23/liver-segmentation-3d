# Предобработка КТ перед подачей в нейросеть
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

# Диапазон КТ, на котором обучалась модель
WINDOW = (-100.0, 200.0)
# Размер входного объёма сети
TARGET_SHAPE = (128, 128, 128)


def resize_volume(vol: np.ndarray, target_shape=TARGET_SHAPE, order: int = 1) -> np.ndarray:
    # Изменить размер объёма до target_shape
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order).astype(np.float32)


def normalize_hu(vol: np.ndarray, window=WINDOW) -> np.ndarray:
    # Клип в диапазон КТ и нормализация в [0, 1]
    lo, hi = window
    return np.clip((np.clip(vol, lo, hi) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def prepare_for_model(hu_zyx: np.ndarray, target_shape=TARGET_SHAPE, window=WINDOW) -> np.ndarray:
    # Полная подготовка (resize + нормализация)
    vol = resize_volume(hu_zyx, target_shape, order=1)
    return normalize_hu(vol, window)
