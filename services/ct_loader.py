# Загрузка КТ-исследований (NIfTI и DICOM)
from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Диапазон КТ, используемый при об]учении
WINDOW: Tuple[float, float] = (-100.0, 200.0)


class CTDataset:
    # Загруженное исследование КТ
    def __init__(self, hu_zyx: np.ndarray, affine: np.ndarray, meta: Dict[str, Any]) -> None:
        self.hu_zyx = hu_zyx
        self.affine = affine
        self.meta = meta

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.hu_zyx.shape)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        # Размер вокселя (z, y, x) в мм
        return (
            float(abs(self.affine[2, 2])),
            float(abs(self.affine[1, 1])),
            float(abs(self.affine[0, 0])),
        )


def load_nifti(path: str | Path) -> CTDataset:
    # Прочитать NIfTI-файл
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)

    # отдаёт (X, Y, Z) -> приводим к (Z, Y, X)
    hu_zyx = np.transpose(data, (2, 1, 0))
    meta = {
        "source": "NIfTI",
        "file_name": Path(path).name,
        "patient_id": Path(path).stem,
        "n_slices": int(hu_zyx.shape[0]),
        "dimensions": list(hu_zyx.shape),
        "hu_min": float(hu_zyx.min()),
        "hu_max": float(hu_zyx.max()),
    }
    return CTDataset(hu_zyx, affine, meta)


def _read_dicom_slice(path: str) -> Optional[Dict[str, Any]]:
    import pydicom

    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception:  
        return None
    if "PixelData" not in ds:
        return None
    if str(getattr(ds, "Modality", "CT")).upper() not in ("CT", ""):
        return None
    try:
        raw = ds.pixel_array
    except Exception:  
        return None
    if raw.ndim != 2:
        return None

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = raw.astype(np.float32) * slope + intercept

    z: Optional[float] = None
    origin = None
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        try:
            origin = [float(v) for v in ipp]
            z = origin[2]
        except Exception:  
            z = None
    if z is None and getattr(ds, "InstanceNumber", None) is not None:
        z = float(ds.InstanceNumber)

    ps = getattr(ds, "PixelSpacing", None)
    row_sp = float(ps[0]) if ps is not None and len(ps) >= 2 else 1.0
    col_sp = float(ps[1]) if ps is not None and len(ps) >= 2 else 1.0

    return {
        "hu": hu,
        "z": z,
        "origin": origin,
        "row_spacing": row_sp,
        "col_spacing": col_sp,
        "thickness": float(getattr(ds, "SliceThickness", 0.0) or 0.0),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
    }


def _find_dicom_files(folder: str) -> List[str]:
    out: List[str] = []
    for root, _dirs, names in os.walk(folder):
        for n in names:
            if n.lower().endswith((".dcm", ".dicom")):
                out.append(os.path.join(root, n))
    if not out:
        skip = (".txt", ".json", ".csv", ".png", ".jpg", ".db", ".nii", ".gz")
        for root, _dirs, names in os.walk(folder):
            for n in names:
                if not n.lower().endswith(skip):
                    out.append(os.path.join(root, n))
    return sorted(out)


def load_dicom(path: str | Path) -> CTDataset:
    # Прочитать папку с DICOM-срезами
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Не найден путь: {path}")

    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.mkdtemp(prefix="liver_dicom_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        folder = tmp
    else:
        folder = str(path)

    slices = [r for f in _find_dicom_files(folder) if (r := _read_dicom_slice(f)) is not None]
    if not slices:
        raise ValueError("Не удалось прочитать ни одного DICOM-среза КТ.")

    # если несколько серий — берём самую большую
    by_series: Dict[str, list] = {}
    for s in slices:
        by_series.setdefault(s["series_uid"], []).append(s)
    if len(by_series) > 1:
        slices = max(by_series.values(), key=len)

    slices.sort(key=lambda s: s["z"] if s["z"] is not None else 0.0)

    shape = slices[0]["hu"].shape
    for s in slices:
        if s["hu"].shape != shape:
            raise ValueError("Срезы разного размера — возможно, смешаны разные серии.")

    # (X, Y, Z) -> (Z, Y, X)
    hu_zyx = np.transpose(np.stack([s["hu"] for s in slices], axis=-1), (2, 1, 0))

    zs = [s["z"] for s in slices if s["z"] is not None]
    if len(zs) >= 2:
        diffs = np.diff(np.sort(np.asarray(zs, dtype=np.float64)))
        diffs = diffs[diffs > 1e-6]
        sz = float(np.median(diffs)) if diffs.size else 1.0
    else:
        sz = slices[0]["thickness"] or 1.0
    sy = float(np.median([s["row_spacing"] for s in slices]))
    sx = float(np.median([s["col_spacing"] for s in slices]))

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = sx
    affine[1, 1] = sy
    affine[2, 2] = sz
    origin = slices[0].get("origin")
    if origin is not None:
        affine[:3, 3] = origin

    meta = {
        "source": "DICOM",
        "file_name": path.name,
        "patient_id": slices[0]["patient_id"] or path.stem,
        "n_slices": len(slices),
        "dimensions": list(hu_zyx.shape),
        "hu_min": float(hu_zyx.min()),
        "hu_max": float(hu_zyx.max()),
    }
    return CTDataset(hu_zyx, affine, meta)


def load_ct(path: str | Path) -> CTDataset:
    # Загрузить КТ из NIfTI или DICOM 
    p = Path(path)
    if p.is_file() and p.suffix.lower() in (".nii", ".gz"):
        return load_nifti(p)
    return load_dicom(p)
