# База данных SQLite (пациенты, исследования, результаты сегментации)
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DB_PATH = Path(os.environ.get("LIVER_DB", "data/liver.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code TEXT,
    created_at   TEXT,
    notes        TEXT
);
CREATE TABLE IF NOT EXISTS studies (
    study_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  INTEGER,
    study_date  TEXT,
    source_file TEXT,
    source_type TEXT,
    dimensions  TEXT,
    spacing     TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
);
CREATE TABLE IF NOT EXISTS segmentation_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    study_id        INTEGER,
    model_version   TEXT,
    volume_ml       REAL,
    voxel_count     INTEGER,
    processing_time REAL,
    mask_path       TEXT,
    confidence      REAL DEFAULT 0.0,
    created_at      TEXT,
    FOREIGN KEY (study_id) REFERENCES studies (study_id)
);
CREATE INDEX IF NOT EXISTS idx_studies_patient ON studies (patient_id);
CREATE INDEX IF NOT EXISTS idx_results_study ON segmentation_results (study_id);
CREATE INDEX IF NOT EXISTS idx_results_created ON segmentation_results (created_at);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> bool:
    # Создать таблицы, если их ещё нет, и выполнить миграцию существующих БД
    try:
        conn = _connect()
        conn.executescript(_SCHEMA)
        # Миграция: добавить колонки, если их нет
        cols = [r[1] for r in conn.execute("PRAGMA table_info(studies)").fetchall()]
        if "source_type" not in cols:
            conn.execute("ALTER TABLE studies ADD COLUMN source_type TEXT")
        cols_r = [r[1] for r in conn.execute("PRAGMA table_info(segmentation_results)").fetchall()]
        if "confidence" not in cols_r:
            conn.execute("ALTER TABLE segmentation_results ADD COLUMN confidence REAL DEFAULT 0.0")
        conn.commit()
        conn.close()
        log.info("База данных: %s", DB_PATH)
        return True
    except Exception as exc:
        log.warning("База данных недоступна: %s", exc)
        return False


def add_patient(patient_code: str, notes: str = "") -> Optional[int]:
    # Добавить пациента
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO patients (patient_code, created_at, notes) VALUES (?, ?, ?)",
        (patient_code, datetime.now().isoformat(timespec="seconds"), notes),
    )
    conn.commit()
    pid = int(cur.lastrowid)
    conn.close()
    return pid


def get_or_create_patient(patient_code: str) -> int:
    # Найти пациента по коду или создать нового
    conn = _connect()
    row = conn.execute(
        "SELECT patient_id FROM patients WHERE patient_code = ?", (patient_code,)
    ).fetchone()
    if row:
        pid = int(row["patient_id"])
    else:
        cur = conn.execute(
            "INSERT INTO patients (patient_code, created_at) VALUES (?, ?)",
            (patient_code, datetime.now().isoformat(timespec="seconds")),
        )
        pid = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return pid


def add_study(patient_id: int, source_file: str, dimensions: str, spacing: str,
              study_date: str = "", source_type: str = "") -> int:
    # Добавить исследование
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO studies (patient_id, study_date, source_file, source_type, dimensions, spacing) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (patient_id, study_date or datetime.now().isoformat(timespec="seconds"),
         source_file, source_type, dimensions, spacing),
    )
    conn.commit()
    sid = int(cur.lastrowid)
    conn.close()
    return sid


def get_or_create_study(patient_code: str, source_file: str, dimensions: str, spacing: str,
                        source_type: str = "") -> int:
    # Найти существующее исследование по пациенту + файлу или создать новое
    pid = get_or_create_patient(patient_code)
    conn = _connect()
    row = conn.execute(
        "SELECT study_id FROM studies WHERE patient_id = ? AND source_file = ? "
        "ORDER BY study_id DESC LIMIT 1",
        (pid, source_file),
    ).fetchone()
    if row:
        sid = int(row["study_id"])
        conn.close()
        return sid
    sid = add_study(pid, source_file, dimensions, spacing, source_type=source_type)
    conn.close()
    return sid


def add_or_update_result(study_id: int, model_version: str, volume_ml: float,
                         voxel_count: int, processing_time: float,
                         mask_path: str = "", confidence: float = 0.0) -> int:
    # Сохранить результат сегментации: обновить существующий для исследования или создать новый
    conn = _connect()
    row = conn.execute(
        "SELECT result_id FROM segmentation_results WHERE study_id = ? "
        "ORDER BY result_id DESC LIMIT 1",
        (study_id,),
    ).fetchone()
    if row:
        rid = int(row["result_id"])
        conn.execute(
            "UPDATE segmentation_results SET model_version = ?, volume_ml = ?, "
            "voxel_count = ?, processing_time = ?, mask_path = ?, confidence = ?, "
            "created_at = ? WHERE result_id = ?",
            (model_version, volume_ml, voxel_count, processing_time, mask_path,
             confidence, datetime.now().isoformat(timespec="seconds"), rid),
        )
        conn.commit()
        conn.close()
        return rid
    cur = conn.execute(
        "INSERT INTO segmentation_results "
        "(study_id, model_version, volume_ml, voxel_count, processing_time, mask_path, confidence, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (study_id, model_version, volume_ml, voxel_count, processing_time,
         mask_path, confidence, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid


def add_result(study_id: int, model_version: str, volume_ml: float, voxel_count: int,
               processing_time: float, mask_path: str = "") -> int:
    # Сохранить результат сегментации
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO segmentation_results "
        "(study_id, model_version, volume_ml, voxel_count, processing_time, mask_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (study_id, model_version, volume_ml, voxel_count, processing_time,
         mask_path, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid


# Разрешённые поля сортировки
_SORTS = {
    "created_at": "r.created_at",
    "volume_ml": "r.volume_ml",
    "voxel_count": "r.voxel_count",
    "processing_time": "r.processing_time",
    "patient_code": "p.patient_code",
}


def search_history(patient_code: str = "", limit: int = 100,
                   order: str = "DESC", sort: str = "created_at") -> List[Dict[str, Any]]:
    # История с поиском по коду пациента и сортировкой
    order = "DESC" if order.upper() == "DESC" else "ASC"
    col = _SORTS.get(sort, "r.created_at")
    conn = _connect()
    sql = (
        "SELECT r.result_id, r.volume_ml, r.voxel_count, "
        "       r.processing_time, r.confidence, r.model_version, r.created_at, "
        "       s.study_id, s.source_file, s.source_type, s.dimensions, s.spacing, "
        "       p.patient_code "
        "FROM segmentation_results r "
        "JOIN studies s ON r.study_id = s.study_id "
        "JOIN patients p ON s.patient_id = p.patient_id "
    )
    params: list = []
    if patient_code:
        sql += "WHERE p.patient_code LIKE ? "
        params.append(f"%{patient_code}%")
    sql += f"ORDER BY {col} {order} LIMIT ?"
    params.append(int(limit))

    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def get_result(result_id: int) -> Optional[Dict[str, Any]]:
    # Получить результат по идентификатору
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM segmentation_results WHERE result_id = ?", (int(result_id),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_result(result_id: int) -> bool:
    # Удалить запись о результате
    conn = _connect()
    conn.execute("DELETE FROM segmentation_results WHERE result_id = ?", (int(result_id),))
    conn.commit()
    conn.close()
    return True


def delete_all_results() -> bool:
    # Удалить все результаты и сбросить счетчик ID на 1
    conn = _connect()
    conn.execute("DELETE FROM segmentation_results")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='segmentation_results'")
    conn.commit()
    conn.close()
    return True


def get_stats() -> Dict[str, Any]:
    # Сводная статистика
    conn = _connect()
    row = conn.execute(
        "SELECT COUNT(*) AS n, ROUND(AVG(volume_ml), 1) AS avg_vol, "
        "ROUND(MIN(volume_ml), 1) AS min_vol, ROUND(MAX(volume_ml), 1) AS max_vol "
        "FROM segmentation_results"
    ).fetchone()
    conn.close()
    d = dict(row) if row else {}
    return {
        "count": d.get("n", 0) or 0,
        "avg_volume_ml": d.get("avg_vol"),
        "min_volume_ml": d.get("min_vol"),
        "max_volume_ml": d.get("max_vol"),
    }
