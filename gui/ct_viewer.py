# Просмотр КТ (три плоскости)
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

MASK_COLOR = (255, 60, 60)
CROSS_COLOR = (0, 200, 255)


class SliceView(QGraphicsView):
    # Просмотр одного среза (КТ + маска + линии пересечения)
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hu: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._cross_v: Optional[int] = None   # вертикальная линия
        self._cross_h: Optional[int] = None   # горизонтальная линия
        self._wc: float = 75.0                # окно отображения КТ (центр)
        self._ww: float = 350.0               # окно отображения КТ (ширина)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base = QGraphicsPixmapItem()
        self._overlay = QGraphicsPixmapItem()
        self._scene.addItem(self._base)
        self._scene.addItem(self._overlay)
        self._overlay.setZValue(1)

        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor(238, 241, 246))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)
        self.setFocusPolicy(Qt.StrongFocus)   # Чтобы колесико мыши работало
        self._fitted = False

    def set_slice(self, hu_slice: np.ndarray, mask_slice: Optional[np.ndarray]) -> None:
        self._hu = np.asarray(hu_slice)
        self._mask = None if mask_slice is None else np.asarray(mask_slice)
        h, w = self._hu.shape
        self._scene.setSceneRect(0, 0, w, h)
        self._render()

    def reset_view(self) -> None:
        # Подогнать изображение под размер окна
        self.resetTransform()
        rect = self._scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)
        self._fitted = True

    def set_cross(self, vertical: Optional[int], horizontal: Optional[int]) -> None:
        self._cross_v = vertical
        self._cross_h = horizontal
        self._render()

    def set_window(self, center: float, width: float) -> None:
        # Изменить окно отображения КТ (HU)
        self._wc = float(center)
        self._ww = max(1.0, float(width))
        self._render()

    def _render(self) -> None:
        if self._hu is None or self._hu.size == 0:
            return
        h, w = self._hu.shape
        lo = self._wc - self._ww / 2.0
        hi = self._wc + self._ww / 2.0
        gray = (np.clip(self._hu, lo, hi) - lo) / max(self._ww, 1.0)
        rgb = (np.clip(gray, 0, 1) * 255).round().astype(np.uint8)
        img = np.stack([rgb, rgb, rgb], axis=-1)

        if self._mask is not None and self._mask.shape == (h, w):
            m = self._mask.astype(bool)
            tint = np.zeros_like(img)
            tint[m] = MASK_COLOR
            img = np.clip(img.astype(np.int16) * 0.65 + tint * 0.35, 0, 255).astype(np.uint8)

        if self._cross_v is not None and 0 <= self._cross_v < w:
            img[:, self._cross_v] = CROSS_COLOR
        if self._cross_h is not None and 0 <= self._cross_h < h:
            img[self._cross_h, :] = CROSS_COLOR

        qimg = QImage(img.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
        self._base.setPixmap(QPixmap.fromImage(qimg.copy()))

    def wheelEvent(self, event) -> None:
        # Колесико мыши: вращение вверх - увеличение, вниз - уменьшение
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        # Сохраняем позицию курсора до масштабирования
        old_pos = self.mapToScene(event.pos())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._panning = True
            self.setCursor(Qt.ClosedHandCursor)
            self._pan_last = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, "_panning", False):
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class OrthoViewer(QWidget):
    # Три плоскости просмотра
    position_changed = pyqtSignal(int, int, int)   # z, y, x

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._hu: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._spacing: Optional[tuple] = None
        self._z = self._y = self._x = 0
        self._linked = True

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(100, 100)

        # три плоскости в ряд, одного размера
        views = QHBoxLayout()
        self.view_axial = SliceView()
        self.view_coronal = SliceView()
        self.view_sagittal = SliceView()
        views.addWidget(self._wrap(self.view_axial, "Аксиальная"), 1)
        views.addWidget(self._wrap(self.view_coronal, "Корональная"), 1)
        views.addWidget(self._wrap(self.view_sagittal, "Сагиттальная"), 1)
        lay.addLayout(views, 1)

        # ползунки срезов
        sliders = QHBoxLayout()
        self.sld_z = self._make_slider("Z")
        self.sld_y = self._make_slider("Y")
        self.sld_x = self._make_slider("X")
        sliders.addWidget(self.sld_z[0])
        sliders.addWidget(self.sld_y[0])
        sliders.addWidget(self.sld_x[0])
        lay.addLayout(sliders)

        # окно отображения КТ (центр / ширина в HU)
        win = QHBoxLayout()
        win.addWidget(QLabel("Яркость:"))
        self.sld_wc = QSlider(Qt.Horizontal)
        self.sld_wc.setRange(-1000, 1000)
        self.sld_wc.setValue(75)
        self.sld_wc.valueChanged.connect(self._on_window)
        win.addWidget(self.sld_wc)
        win.addWidget(QLabel("Контраст:"))
        self.sld_ww = QSlider(Qt.Horizontal)
        self.sld_ww.setRange(1, 2000)
        self.sld_ww.setValue(350)
        self.sld_ww.valueChanged.connect(self._on_window)
        win.addWidget(self.sld_ww)
        lay.addLayout(win)

    def _wrap(self, view, title: str) -> QWidget:
        box = QWidget()
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l = QVBoxLayout(box)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        l.addWidget(QLabel(f"<b>{title}</b>"))
        l.addWidget(view, 1)
        return box

    def _make_slider(self, axis: str):
        box = QWidget()
        l = QVBoxLayout(box)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(QLabel(axis))
        sld = QSlider(Qt.Horizontal)
        l.addWidget(sld)
        return box, sld

    def set_volume(self, hu: np.ndarray, mask: Optional[np.ndarray],
                   spacing: Optional[tuple] = None) -> None:
        # Установить объём (Z, Y, X) и шаг вокселя (z, y, x) в мм
        self._hu = hu
        self._mask = mask
        self._spacing = spacing
        d, h, w = hu.shape
        self._z, self._y, self._x = d // 2, h // 2, w // 2

        self.sld_z[1].setRange(0, d - 1)
        self.sld_y[1].setRange(0, h - 1)
        self.sld_x[1].setRange(0, w - 1)
        self.sld_z[1].setValue(self._z)
        self.sld_y[1].setValue(self._y)
        self.sld_x[1].setValue(self._x)

        self.sld_z[1].valueChanged.connect(lambda v: self._on_change("z", v))
        self.sld_y[1].valueChanged.connect(lambda v: self._on_change("y", v))
        self.sld_x[1].valueChanged.connect(lambda v: self._on_change("x", v))
        self._refresh()
        # подогнать срезы под размер окон, чтобы не было пустого фона
        self.view_axial.reset_view()
        self.view_coronal.reset_view()
        self.view_sagittal.reset_view()

    def _on_change(self, axis: str, value: int) -> None:
        if axis == "z":
            self._z = value
        elif axis == "y":
            self._y = value
        else:
            self._x = value
        self._refresh()

    def _on_window(self) -> None:
        wc = float(self.sld_wc.value())
        ww = float(self.sld_ww.value())
        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.set_window(wc, ww)
        self._refresh()

    def _refresh(self) -> None:
        if self._hu is None:
            return
        hu, mask = self._hu, self._mask
        z, y, x = self._z, self._y, self._x

        # аксиальная: hu[z] (Y, X); линии: y (горизонталь), x (вертикаль)
        m = None if mask is None else mask[z]
        self.view_axial.set_slice(hu[z], m)
        self.view_axial.set_cross(vertical=x, horizontal=y)

        # корональная: hu[:, y, :] (Z, X); линии: z (горизонталь), x (вертикаль)
        m = None if mask is None else mask[:, y, :]
        self.view_coronal.set_slice(hu[:, y, :], m)
        self.view_coronal.set_cross(vertical=x, horizontal=z)

        # сагиттальная: hu[:, :, x] (Z, Y); линии: z (горизонталь), y (вертикаль)
        m = None if mask is None else mask[:, :, x]
        self.view_sagittal.set_slice(hu[:, :, x], m)
        self.view_sagittal.set_cross(vertical=y, horizontal=z)

        self.position_changed.emit(z, y, x)

    def set_position(self, z: int, y: int, x: int) -> None:
        if self._hu is None:
            return
        d, h, w = self._hu.shape
        self._z = int(np.clip(z, 0, d - 1))
        self._y = int(np.clip(y, 0, h - 1))
        self._x = int(np.clip(x, 0, w - 1))
        self.sld_z[1].setValue(self._z)
        self.sld_y[1].setValue(self._y)
        self.sld_x[1].setValue(self._x)
        self._refresh()

    def current_position(self):
        return self._z, self._y, self._x