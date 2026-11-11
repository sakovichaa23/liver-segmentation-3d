# Трёхмерный просмотр печени
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from services import volume_calculator


def _isosurface(mask: np.ndarray, spacing: Tuple[float, float, float]):
    # Вершины и грани изоповерхности маски
    from skimage.measure import marching_cubes

    verts, faces, _n, _v = marching_cubes(mask.astype(np.float32), level=0.5)
    return verts * np.asarray(spacing, dtype=np.float32), faces


class VolumeViewer(QWidget):
    # Отображение 3D-модели печени
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mask: Optional[np.ndarray] = None
        self._spacing = (1.0, 1.0, 1.0)
        self._azimuth = 30.0
        self._elevation = 20.0
        self._zoom = 1.0
        self._verts = None
        self._faces = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.lbl = QLabel("Нет 3D-модели")
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setMinimumHeight(260)
        self.lbl.setStyleSheet("background:#101014; color:#7a8794; border-radius:8px;")
        lay.addWidget(self.lbl, 1)

        controls = QVBoxLayout()
        row1 = QVBoxLayout()
        row1.addWidget(QLabel("Поворот"))
        self.sld_az = QSlider(Qt.Horizontal)
        self.sld_az.setRange(0, 360)
        self.sld_az.setValue(int(self._azimuth))
        self.sld_az.valueChanged.connect(lambda v: self._set(azimuth=float(v)))
        row1.addWidget(self.sld_az)
        controls.addLayout(row1)

        row2 = QVBoxLayout()
        row2.addWidget(QLabel("Высота"))
        self.sld_el = QSlider(Qt.Horizontal)
        self.sld_el.setRange(-90, 90)
        self.sld_el.setValue(int(self._elevation))
        self.sld_el.valueChanged.connect(lambda v: self._set(elevation=float(v)))
        row2.addWidget(self.sld_el)
        controls.addLayout(row2)

        row3 = QVBoxLayout()
        row3.addWidget(QLabel("Масштаб"))
        self.sld_zoom = QSlider(Qt.Horizontal)
        self.sld_zoom.setRange(50, 200)
        self.sld_zoom.setValue(100)
        self.sld_zoom.valueChanged.connect(lambda v: self._set(zoom=v / 100.0))
        row3.addWidget(self.sld_zoom)
        controls.addLayout(row3)
        lay.addLayout(controls)

        self.btn_window = QPushButton("Открыть в отдельном окне")
        self.btn_window.clicked.connect(self.open_interactive)
        lay.addWidget(self.btn_window)

    def set_mask(self, mask: np.ndarray, spacing: Tuple[float, float, float]) -> None:
        self._mask = mask
        self._spacing = spacing
        try:
            self._verts, self._faces = _isosurface(mask, spacing)
        except Exception:  
            self._verts = self._faces = None
        self._render()

    def clear(self) -> None:
        self._mask = None
        self._verts = self._faces = None
        self.lbl.setText("Нет 3D-модели")
        self.lbl.setPixmap(QPixmap())

    def _set(self, azimuth=None, elevation=None, zoom=None) -> None:
        if azimuth is not None:
            self._azimuth = azimuth
        if elevation is not None:
            self._elevation = elevation
        if zoom is not None:
            self._zoom = zoom
        self._render()

    def _render(self) -> None:
        if self._verts is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(4.2, 3.2), dpi=110)
            ax = fig.add_subplot(111, projection="3d")
            mesh = Poly3DCollection(
                self._verts[self._faces], alpha=0.9, facecolor="#c0392b", edgecolor="#7b241c"
            )
            ax.add_collection3d(mesh)

            lims = self._verts.max(axis=0) - self._verts.min(axis=0)
            center = (self._verts.max(axis=0) + self._verts.min(axis=0)) / 2
            r = float(lims.max()) / 2 * self._zoom
            ax.set_xlim(center[0] - r, center[0] + r)
            ax.set_ylim(center[1] - r, center[1] + r)
            ax.set_zlim(center[2] - r, center[2] + r)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=self._elevation, azim=self._azimuth)
            ax.set_axis_off()
            fig.tight_layout(pad=0)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.asarray(fig.canvas.buffer_rgba())
            buf = buf[:, :, :3]  # RGBA -> RGB
            plt.close(fig)

            img = QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
            self.lbl.setPixmap(QPixmap.fromImage(img.copy()).scaled(
                self.lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as exc: 
            self.lbl.setText(f"Ошибка 3D: {exc}")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._verts is not None:
            self._render()

    def open_interactive(self) -> None:
        if self._mask is None:
            return
        try:
            import vedo

            mesh = vedo.Mesh([self._verts, self._faces]).color("tomato").lw(0.5)
            plt = vedo.Plotter(title="Печень — 3D", axes=1, bg="white")
            plt.show(mesh, interactive=True)
            plt.close()
        except ImportError:
            self.lbl.setText("vedo не установлен — используйте ползунки выше")