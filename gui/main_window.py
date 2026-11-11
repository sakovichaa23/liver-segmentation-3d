"""Главное окно приложения
Функциональность:
  * загрузка КТ: NIfTI или DICOM;
  * автоматическая 3D-сегментация печени обученной U-Net;
  * просмотр в трёх ортогональных плоскостях с перекрёстной навигацией;
  * 3D-модель печени (в отдельном окне по кнопке);
  * расчёт объёма в мл и числа вокселей;
  * экспорт маски в NIfTI и DICOM, PDF-отчёт;
  * история исследований (SQLite) с поиском и сортировкой.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch  # импортируется ДО PyQt5
import yaml
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from database import db
from gui.ct_viewer import OrthoViewer
from gui.history_window import HistoryWindow
from services import ct_loader, segmentation
from services import report_generator, volume_calculator

log = logging.getLogger("gui")

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

STYLESHEET = """
QMainWindow, QWidget { background: #f4f6fa; color: #22303f; font-family: "Segoe UI", Arial; font-size: 18px; }
QLabel { color: #22303f; }
QLabel#volume { font-size: 52px; font-weight: 800; color: #1d4e89; padding: 2px; }
QLabel#caption { font-size: 22px; color: #7a8794; }
QLabel#panel { font-size: 21px; font-weight: 700; color: #1d4e89; padding: 4px 0 6px 0; }
QFrame#card { background: transparent; border: none; }
QFrame#vline { border-left: 1.5px solid #c9d4e0; background: transparent; }
QToolBar { background: #ffffff; border-bottom: 1px solid #dde3ea; padding: 8px; spacing: 8px; }
QPushButton { background: #2f6fb5; color: #ffffff; border: none; border-radius: 6px; padding: 10px 24px; font-weight: 600; font-size: 18px; }
QPushButton:hover { background: #285f9c; }
QPushButton:pressed { background: #1f4d80; }
QPushButton:disabled { background: #c3cdda; color: #f0f2f5; }
QPushButton#secondary { background: #e8edf4; color: #22303f; }
QPushButton#secondary:hover { background: #d9e1ec; }
QPushButton#ghost { background: transparent; color: #2f6fb5; border: 1px solid #b9cde4; }
QPushButton#ghost:hover { background: #eef4fb; }
QProgressBar { border: 1px solid #c9d4e0; border-radius: 6px; text-align: center; height: 22px; background: #ffffff; font-size: 16px; }
QProgressBar::chunk { background: #2f6fb5; border-radius: 6px; }
QTextEdit { background: #ffffff; border: 1px solid #dde3ea; border-radius: 8px; font-size: 18px; }
QMenuBar { background: #ffffff; border-bottom: 1px solid #dde3ea; font-size: 18px; }
QMenuBar::item:selected { background: #e8f0fa; }
QMenu { background: #ffffff; border: 1px solid #dde3ea; font-size: 18px; }
QMenu::item:selected { background: #e8f0fa; }
QStatusBar { background: #ffffff; border-top: 1px solid #dde3ea; color: #7a8794; font-size: 17px; }
QTableWidget { background: #ffffff; border: 1px solid #dde3ea; gridline-color: #e6ebf1; font-size: 18px; }
QHeaderView::section { background: #eef2f7; color: #22303f; font-size: 17px; font-weight: 600; padding: 6px; border: none; border-bottom: 1px solid #dde3ea; }
QComboBox { background: #ffffff; border: 1px solid #c9d4e0; border-radius: 6px; padding: 6px 10px; font-size: 18px; }
QLineEdit { background: #ffffff; border: 1px solid #c9d4e0; border-radius: 6px; padding: 6px 10px; font-size: 18px; }
QSlider::handle:horizontal { background: #2f6fb5; width: 16px; height: 16px; border-radius: 8px; margin: -3px 0; }
QSlider::groove:horizontal { background: #c9d4e0; height: 5px; border-radius: 2px; }
"""


class Model3DDialog(QDialog):
    def __init__(self, verts: np.ndarray, faces: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("3D-модель печени")
        self.resize(800, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        ax = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces], alpha=0.9, facecolor="#8b2626", edgecolor="#521717")
        ax.add_collection3d(mesh)
        lims = verts.max(axis=0) - verts.min(axis=0)
        center = (verts.max(axis=0) + verts.min(axis=0)) / 2
        r = float(lims.max()) / 2
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=30)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        self.canvas.draw()


class SegmentationThread(QThread):
    # Фоновая сегментация
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, net, dataset, device: torch.device, model_version: str) -> None:
        super().__init__()
        self.net = net
        self.dataset = dataset
        self.device = device
        self.model_version = model_version

    def run(self) -> None:
        try:
            t0 = time.time()
            result = segmentation.segment(
                self.net, self.dataset.hu_zyx, self.dataset.affine, self.device,
                progress=lambda p: self.progress.emit(int(p)),
            )
            result.processing_time = time.time() - t0
            result.model_version = self.model_version
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("Ошибка сегментации")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Сегментация печени по КТ")
        self.resize(1400, 900)

        self.net = None
        self.device = segmentation.resolve_device(CFG["model"].get("device", "auto"))
        self.dataset = None
        self.mask: Optional[np.ndarray] = None
        self.result = None
        self.current_study_id: Optional[int] = None
        self.current_patient_code = ""
        self.current_source_type: str = ""
        self.thread: Optional[SegmentationThread] = None
        self._last_position: tuple = (0, 0, 0)

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._load_model()
        self._init_db()

    # UI
    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # три просмотра на всю ширину (максимальный размер изображений)
        self.ortho = OrthoViewer()
        self.ortho.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ortho_card = self._card("Просмотр срезов", self.ortho, stretch=1)
        self.ortho_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ortho_card.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ortho_card, 1)

        # нижняя панель: три колонки
        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        bottom.setContentsMargins(4, 4, 4, 4)

        # левая колонка — данные исследования, позиция, площадь среза
        left_panel = QHBoxLayout()
        left_panel.setSpacing(10)
        left_panel.setContentsMargins(0, 0, 0, 0)

        col_info = QVBoxLayout()
        col_info.setSpacing(4)
        col_info.setContentsMargins(0, 0, 0, 0)
        self.lbl_info = QLabel("Загрузите КТ (NIfTI / DICOM).")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_info.setStyleSheet("color:#3a4b5c; padding:4px; font-size: 17px; line-height: 1.4;")
        col_info.addWidget(self._card("Данные исследования", self.lbl_info))
        left_panel.addLayout(col_info, 1)

        col_pos_area = QVBoxLayout()
        col_pos_area.setSpacing(4)
        col_pos_area.setContentsMargins(0, 0, 0, 0)

        self.lbl_axis = QLabel("Ось Z: —\nОсь Y: —\nОсь X: —")
        self.lbl_axis.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_axis.setStyleSheet("color:#3a4b5c; padding:4px; font-size: 17px; line-height: 1.4;")
        col_pos_area.addWidget(self._card("Текущая позиция", self.lbl_axis))

        self.lbl_area = QLabel("Площадь среза:\n—")
        self.lbl_area.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_area.setStyleSheet("color:#3a4b5c; padding:4px; font-size: 17px; line-height: 1.4;")
        col_pos_area.addWidget(self._card("Площадь среза", self.lbl_area))
        left_panel.addLayout(col_pos_area, 1)

        bottom.addLayout(left_panel, 1)

        # средняя колонка — 3D модель печени (отображается автоматически после сегментации)
        middle_panel = QVBoxLayout()
        middle_panel.setSpacing(0)
        middle_panel.setContentsMargins(0, 0, 0, 0)

        self.lbl_3d = QLabel("Результат появится здесь после сегментации")
        self.lbl_3d.setAlignment(Qt.AlignCenter)
        self.lbl_3d.setStyleSheet("color:#5b6b7c; background:#ffffff; border:1px solid #dde3ea; border-radius:8px;")
        self.lbl_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        middle_panel.addWidget(self.lbl_3d)

        bottom.addLayout(middle_panel, 1)

        # правая колонка — результаты измерений
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.addStretch(1)

        self.lbl_volume = QLabel("—", objectName="volume")
        self.lbl_volume.setAlignment(Qt.AlignCenter)
        self.lbl_caption = QLabel("Объём печени", objectName="caption")
        self.lbl_caption.setAlignment(Qt.AlignCenter)
        vol_card = QFrame(objectName="card")
        vol_lay = QVBoxLayout(vol_card)
        vol_lay.setContentsMargins(0, 0, 0, 0)
        vol_lay.setSpacing(1)
        vol_lay.addWidget(self.lbl_volume)
        vol_lay.addWidget(self.lbl_caption)
        right_panel.addWidget(vol_card)

        # кнопка "Запустить в отдельном окне"
        self.btn_3d = QPushButton("▶  Открыть в отдельном окне 3D модель печени")
        self.btn_3d.setEnabled(False)
        self.btn_3d.clicked.connect(self.open_3d_window)
        right_panel.addWidget(self.btn_3d)

        self.lbl_metrics = QLabel("Воксели: —   |   Время: — с   |   Уверенность: — %")
        self.lbl_metrics.setAlignment(Qt.AlignCenter)
        self.lbl_metrics.setStyleSheet("color:#5b6b7c; padding:2px; font-size: 16px;")
        right_panel.addWidget(self.lbl_metrics)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setMaximumHeight(14)
        right_panel.addWidget(self.progress)
        right_panel.addStretch(1)

        bottom.addLayout(right_panel, 1)

        layout.addLayout(bottom, 0)

        # изображения растягиваются при полном экране, нижняя панель фиксированная
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # связь позиции срезов с обновлением оси и площади среза
        self.ortho.position_changed.connect(self._on_position_changed)

    def _card(self, title: str, body: QWidget, stretch: int = 0) -> QFrame:
        card = QFrame(objectName="card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        lay.addWidget(QLabel(title, objectName="panel"))
        lay.addWidget(body, stretch)
        return card

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")

        act_nifti = QAction("Загрузить КТ (NIfTI)…", self)
        act_nifti.setShortcut(QKeySequence.Open)
        act_nifti.triggered.connect(self.load_nifti)
        file_menu.addAction(act_nifti)

        act_dicom = QAction("Загрузить КТ (DICOM)…", self)
        act_dicom.triggered.connect(self.load_dicom)
        file_menu.addAction(act_dicom)

        file_menu.addSeparator()

        act_mask_nii = QAction("Сохранить маску (NIfTI)…", self)
        act_mask_nii.triggered.connect(self.export_mask_nifti)
        file_menu.addAction(act_mask_nii)

        act_mask_dcm = QAction("Сохранить маску (DICOM)…", self)
        act_mask_dcm.triggered.connect(self.export_mask_dicom)
        file_menu.addAction(act_mask_dcm)

        act_pdf = QAction("PDF-отчёт…", self)
        act_pdf.triggered.connect(self.export_pdf)
        file_menu.addAction(act_pdf)

        file_menu.addSeparator()
        act_quit = QAction("Выход", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        tools_menu = menubar.addMenu("Инструменты")
        act_hist = QAction("История исследований", self)
        act_hist.triggered.connect(self.show_history)
        tools_menu.addAction(act_hist)

        act_reset = QAction("Сбросить просмотр", self)
        act_reset.triggered.connect(self.reset_view)
        tools_menu.addAction(act_reset)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Инструменты")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.btn_segment = QPushButton("▶  Запустить сегментацию")
        self.btn_segment.clicked.connect(self.run_segmentation)
        tb.addWidget(self.btn_segment)

    # модель
    def _load_model(self) -> None:
        weights = ROOT / CFG["model"]["weights"]
        if not weights.exists():
            self.status.showMessage(f"Веса не найдены: {weights}", 8000)
            return
        try:
            self.net = segmentation.load_model(weights, self.device)
            self.model_version = weights.name
            self.status.showMessage(f"Модель загружена ({self.device})", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка модели", str(exc))

    def _init_db(self) -> None:
        db.DB_PATH = ROOT / str(CFG.get("database", {}).get("path", "data/liver.db"))
        db.init_db()

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait(1000)
        event.accept()

    # загрузка
    def load_nifti(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите КТ (NIfTI)", "", "NIfTI (*.nii *.nii.gz);;Все файлы (*)"
        )
        if paths:
            self.load_ct(paths[0])

    def load_dicom(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите DICOM-срезы")
        if path:
            self.load_ct(path)

    def load_ct(self, path: str) -> None:
        try:
            self.dataset = ct_loader.load_ct(path)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка загрузки", str(exc))
            return

        self.current_patient_code = self.dataset.meta.get("patient_id") or Path(path).stem
        self.current_source_type = self.dataset.meta.get("source_type", "NIfTI" if path.lower().endswith((".nii", ".nii.gz")) else "DICOM")
        self.mask = None
        self.result = None
        self.progress.setValue(0)

        self.ortho.set_volume(self.dataset.hu_zyx, None, self.dataset.spacing)
        self._update_info()
        self.lbl_volume.setText("—")
        self.lbl_metrics.setText("Воксели: —")
        self.lbl_3d.setText("Результат появится здесь после сегментации")
        self.lbl_3d.setPixmap(QPixmap())
        self.btn_segment.setEnabled(True)
        self.btn_3d.setEnabled(False)
        self.status.showMessage(f"Загружено: {self.dataset.meta['file_name']}", 4000)

        # сохранить пациента и исследование в базе (без дублей)
        try:
            spacing = self.dataset.spacing
            self.current_study_id = db.get_or_create_study(
                self.current_patient_code, self.dataset.meta["file_name"],
                "×".join(str(s) for s in self.dataset.hu_zyx.shape),
                " × ".join(f"{s:.2f}" for s in spacing),
                source_type=self.current_source_type,
            )
        except Exception as exc:
            log.warning("Не удалось сохранить исследование: %s", exc)

    def _update_info(self) -> None:
        if self.dataset is None:
            return
        s = self.dataset.spacing
        self.lbl_info.setText(
            f"Пациент: {self.current_patient_code}\n"
            f"Файл: {self.dataset.meta['file_name']}\n"
            f"Формат: {self.current_source_type}\n"
            f"Размер: {self.dataset.hu_zyx.shape}\n"
            f"Срезов: {self.dataset.hu_zyx.shape[0]}\n"
            f"Шаг: {s[2]:.1f}×{s[1]:.1f}×{s[0]:.1f} мм"
        )

    def _on_position_changed(self, z: int, y: int, x: int) -> None:
        # Обновить отображение текущей позиции осей и площади среза
        self.lbl_axis.setText(f"Ось Z: {z + 1}\nОсь Y: {y + 1}\nОсь X: {x + 1}")
        self._last_position = (z, y, x)

        # площадь среза (только если есть маска и интервал)
        if self.mask is not None and self.dataset is not None:
            sy, sx = self.dataset.spacing[1], self.dataset.spacing[2]
            area_mm2 = int(self.mask[z].sum()) * sy * sx
            self.lbl_area.setText(f"Площадь среза:\n{area_mm2 / 100.0:.1f} см²")
            self._update_3d_preview()
        else:
            self.lbl_area.setText("Площадь среза:\n—")

    # сегментация
    def run_segmentation(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "Нет данных", "Сначала загрузите КТ.")
            return
        if self.net is None:
            QMessageBox.warning(self, "Нет модели", "Модель не загружена.")
            return

        self.btn_segment.setEnabled(False)
        self.progress.setValue(0)
        self.status.showMessage("Сегментация…")

        self.thread = SegmentationThread(self.net, self.dataset, self.device, self.model_version)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished_ok.connect(self.on_segmentation_done)
        self.thread.failed.connect(self.on_segmentation_failed)
        self.thread.start()

    def on_segmentation_done(self, result) -> None:
        self.btn_segment.setEnabled(True)
        self.mask = result.mask
        self.result = result
        self.ortho.set_volume(self.dataset.hu_zyx, self.mask, self.dataset.spacing)
        self.lbl_volume.setText(f"{result.volume_ml:,.1f} мл")
        self.status.showMessage(f"Готово: {result.volume_ml:,.1f} мл", 6000)

        mask_path = ""
        try:
            db.add_or_update_result(
                self.current_study_id, self.model_version, result.volume_ml,
                result.voxel_count, result.processing_time, mask_path,
                confidence=result.confidence,
            )
        except Exception as exc:
            log.warning("Не удалось сохранить результат: %s", exc)

        self.lbl_metrics.setText(
            f"Вокселей печени: {result.voxel_count:,}   |   "
            f"Время обработки: {result.processing_time:.2f} с   |   "
            f"Уверенность: {result.confidence:.1f} %"
        )

        self.btn_3d.setEnabled(True)
        # Показать 3D модель сразу после сегментации
        self._update_3d_preview()

    def on_segmentation_failed(self, message: str) -> None:
        self.btn_segment.setEnabled(True)
        self.progress.setValue(0)
        QMessageBox.critical(self, "Ошибка сегментации", message)

    # экспорт
    def _require_mask(self) -> bool:
        if self.mask is None:
            QMessageBox.information(self, "Нет маски", "Сначала выполните сегментацию.")
            return False
        return True

    def export_mask_nifti(self) -> None:
        if not self._require_mask():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить маску", "liver_mask.nii.gz", "NIfTI (*.nii.gz)")
        if not path:
            return
        import nibabel as nib

        img = nib.Nifti1Image(np.transpose(self.mask, (1, 2, 0)).astype(np.uint8), self.dataset.affine)
        nib.save(img, path)
        self.status.showMessage(f"Маска сохранена: {path}", 5000)

    def export_mask_dicom(self) -> None:
        if not self._require_mask():
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка для DICOM-маски")
        if not folder:
            return
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileMetaDataset
            from pydicom.uid import ExplicitVRLittleEndian, generate_uid

            series_uid = generate_uid()
            for z in range(self.mask.shape[0]):
                ds = Dataset()
                ds.file_meta = FileMetaDataset()
                ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
                ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                ds.Modality = "CT"
                ds.PatientID = self.current_patient_code
                ds.Rows, ds.Columns = self.mask.shape[1], self.mask.shape[2]
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.SeriesInstanceUID = series_uid
                ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
                ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
                ds.PixelData = (self.mask[z] * 255).astype(np.uint16).tobytes()
                ds.save_as(str(Path(folder) / f"mask_{z:04d}.dcm"), enforce_file_format=True)
            self.status.showMessage(f"DICOM-маска сохранена: {folder}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))

    def export_pdf(self) -> None:
        if not self._require_mask():
            return
        path, _ = QFileDialog.getSaveFileName(self, "PDF-отчёт", "report.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            rgb = self._make_report_image()
            s = self.dataset.spacing
            params = {
                "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "n_slices": self.dataset.hu_zyx.shape[0],
                "image_shape": f"{self.dataset.hu_zyx.shape[1]}×{self.dataset.hu_zyx.shape[2]}",
                "spacing": s,
                "voxel_mm3": volume_calculator.voxel_volume_mm3(self.dataset.affine),
                "liver_voxels": int(self.mask.sum()),
            }
            report_generator.make_report(
                path, self.current_patient_code, self.dataset.meta["file_name"], rgb, params,
                self.result.volume_ml, model_version=self.model_version,
                processing_time=self.result.processing_time,
            )
            self.status.showMessage(f"PDF сохранён: {path}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка PDF", str(exc))

    @staticmethod
    def _overlay_rgb(hu_slice: np.ndarray, mask_slice: np.ndarray, window) -> np.ndarray:
        import cv2

        lo, hi = window
        gray = (np.clip((np.clip(hu_slice, lo, hi) - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        m = mask_slice.astype(bool)
        if m.any():
            tint = np.zeros_like(rgb)
            tint[m] = (255, 60, 60)
            rgb = cv2.addWeighted(rgb, 1.0, tint, 0.4, 0)
            contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 60, 60), 2)
        return rgb

    def _render_3d_image(self, width: int = 600, height: int = 500) -> Optional[np.ndarray]:
        # Отрендерить 3D-модель печени в RGB-массив
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from skimage.measure import marching_cubes

            verts, faces, _n, _v = marching_cubes(self.mask.astype(np.float32), level=0.5)
            verts = verts * np.asarray(self.dataset.spacing, dtype=np.float32)

            fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            mesh = Poly3DCollection(verts[faces], alpha=0.9, facecolor="#8b2626", edgecolor="#521717")
            ax.add_collection3d(mesh)
            lims = verts.max(axis=0) - verts.min(axis=0)
            center = (verts.max(axis=0) + verts.min(axis=0)) / 2
            r = float(lims.max()) / 2
            ax.set_xlim(center[0] - r, center[0] + r)
            ax.set_ylim(center[1] - r, center[1] + r)
            ax.set_zlim(center[2] - r, center[2] + r)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=30)
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)
            return buf
        except Exception:
            return None

    def _update_3d_preview(self) -> None:
        if self.mask is None or self.dataset is None:
            return
        try:
            z = self._last_position[0]
            z = min(max(0, z), self.mask.shape[0] - 1)
            slice_mask = self.mask[z]
            if slice_mask.sum() == 0:
                slice_mask = self.mask.max(axis=0)

            if slice_mask.sum() == 0:
                self.lbl_3d.setPixmap(QPixmap())
                return

            rows = np.any(slice_mask, axis=1)
            cols = np.any(slice_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            cropped = slice_mask[rmin:rmax + 1, cmin:cmax + 1]
            ch, cw = cropped.shape

            pad_y = max(10, int(ch * 0.12))
            pad_x = max(10, int(cw * 0.12))
            padded = np.pad(cropped, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0)
            ph, pw = padded.shape

            sy, sx = self.dataset.spacing[1], self.dataset.spacing[2]
            if sx > 0 and sy > 0:
                new_ph = int(round(ph * (sy / sx)))
                import cv2
                padded = cv2.resize(padded.astype(np.uint8), (pw, new_ph), interpolation=cv2.INTER_NEAREST)
                ph = new_ph

            rgb = np.full((ph, pw, 3), 255, dtype=np.uint8)
            rgb[padded > 0] = [139, 38, 38]

            qimg = QImage(rgb.data, pw, ph, 3 * pw, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())

            lbl_w = max(10, self.lbl_3d.width() - 8)
            lbl_h = max(10, self.lbl_3d.height() - 8)
            self.lbl_3d.setPixmap(pix.scaled(lbl_w, lbl_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.mask is not None:
            self._update_3d_preview()

    def _make_report_image(self) -> np.ndarray:
        # Сводное изображение для отчёта
        import cv2
        from services.preprocessing import WINDOW

        hu = self.dataset.hu_zyx
        mask = self.mask
        z, y, x = self.ortho.current_position()
        z = min(z, hu.shape[0] - 1)
        y = min(y, hu.shape[1] - 1)
        x = min(x, hu.shape[2] - 1)

        planes = [
            self._overlay_rgb(hu[z], mask[z], WINDOW),
            self._overlay_rgb(hu[:, y, :], mask[:, y, :], WINDOW),
            self._overlay_rgb(hu[:, :, x], mask[:, :, x], WINDOW),
        ]

        target_h = 380
        row = []
        for ov in planes:
            h, w = ov.shape[:2]
            scale = min(target_h / h, target_h / w)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            r = cv2.resize(ov, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            y0 = (target_h - nh) // 2
            x0 = (target_h - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = r
            row.append(canvas)
        planes_img = np.concatenate(row, axis=1)

        img3d = self._render_3d_image(width=target_h, height=target_h)
        if img3d is not None:
            img3d = cv2.resize(img3d, (target_h, target_h), interpolation=cv2.INTER_AREA)
            planes_img = np.concatenate([planes_img, img3d], axis=1)

        # подписи
        labels = ["Аксиальная", "Корональная", "Сагиттальная"] + (["3D-модель"] if img3d is not None else [])
        for i, text in enumerate(labels):
            cx = i * target_h + 12
            cv2.putText(planes_img, text, (cx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return planes_img

    # прочее
    def reset_view(self) -> None:
        if self.dataset is not None:
            self.ortho.set_volume(self.dataset.hu_zyx, self.mask, self.dataset.spacing)

    def show_history(self) -> None:
        HistoryWindow(self, on_open=self._open_result).exec_()

    def _open_result(self, result_id: int) -> None:
        row = db.get_result(result_id)
        if not row:
            return
        # получить источник исследования из связанной таблицы studies
        conn = db._connect()
        src = conn.execute(
            "SELECT s.source_file, s.source_type, p.patient_code "
            "FROM segmentation_results r "
            "JOIN studies s ON r.study_id = s.study_id "
            "JOIN patients p ON s.patient_id = p.patient_id "
            "WHERE r.result_id = ?", (int(result_id),)
        ).fetchone()
        conn.close()
        patient_code = src["patient_code"] if src else "—"
        source_file = src["source_file"] if src else "—"
        source_type = src["source_type"] if src else "—"
        conf = row.get("confidence", 0)
        conf_text = f"{round(conf)} %" if conf is not None else "— %"
        QMessageBox.information(
            self, "Результат",
            f"Пациент: {patient_code}\n"
            f"Файл: {source_file}\n"
            f"Формат: {source_type}\n"
            f"Объём: {row['volume_ml']:,.1f} мл\n"
            f"Вокселей: {row['voxel_count']:,}\n"
            f"Модель: {row['model_version']}\n"
            f"Уверенность: {conf_text}\n"
            f"Дата: {row['created_at']}",
        )

    def open_3d_window(self) -> None:
        if self.mask is None or self.dataset is None:
            QMessageBox.information(self, "Нет данных", "Сначала выполните сегментацию.")
            return
        try:
            import vedo
            from skimage.measure import marching_cubes

            verts, faces, _n, _v = marching_cubes(self.mask.astype(np.float32), level=0.5)
            verts = verts * np.asarray(self.dataset.spacing, dtype=np.float32)
            mesh = vedo.Mesh([verts, faces]).color("tomato").lw(0.5)
            plt = vedo.Plotter(title="Печень — 3D", axes=1, bg="white")
            plt.show(mesh, interactive=True)
            plt.close()
            return
        except ImportError:
            pass
        except Exception as exc:
            log.warning("Ошибка vedo: %s", exc)

        try:
            from skimage.measure import marching_cubes

            verts, faces, _n, _v = marching_cubes(self.mask.astype(np.float32), level=0.5)
            verts = verts * np.asarray(self.dataset.spacing, dtype=np.float32)
            dlg = Model3DDialog(verts, faces, self)
            dlg.exec_()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка 3D", str(exc))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Сегментация печени по КТ")
    window = MainWindow()
    window.show()
    import os
    code = app.exec_()
    os._exit(code)


if __name__ == "__main__":
    main()