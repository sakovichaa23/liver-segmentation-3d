# Окно истории исследований (сортировка, удаление, повторное открытие)
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from database import db

# Варианты сортировки (▲ возрастание, ▼ убывание)
_SORT_OPTIONS = [
    ("По дате ▲", "created_at", "ASC"),
    ("По дате ▼", "created_at", "DESC"),
    ("По объёму ▲", "volume_ml", "ASC"),
    ("По объёму ▼", "volume_ml", "DESC"),
    ("По вокселям ▲", "voxel_count", "ASC"),
    ("По вокселям ▼", "voxel_count", "DESC"),
    ("По времени ▲", "processing_time", "ASC"),
    ("По времени ▼", "processing_time", "DESC"),
]

_HEADERS = ["ID", "Пациент", "Объём, мл", "Уверенность, %", "Воксели", "Время, с",
            "Модель", "Дата"]


class HistoryWindow(QDialog):
    # История результатов из БД
    def __init__(self, parent=None, on_open: Optional[Callable[[int], None]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("История исследований")
        self.resize(1080, 560)
        self.on_open = on_open
        self._rows: list = []

        lay = QVBoxLayout(self)

        # панель сортировки (поиск убран)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Сортировка:"))
        self.cmb_order = QComboBox()
        self.cmb_order.addItems([o[0] for o in _SORT_OPTIONS])
        self.cmb_order.currentIndexChanged.connect(lambda _: self.refresh())
        controls.addWidget(self.cmb_order)
        controls.addStretch(1)
        lay.addLayout(controls)

        # таблица
        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)  # ручная сортировка через комбобокс
        lay.addWidget(self.table, 1)

        # всего записей
        self.lbl_count = QLabel("Всего записей: 0")
        self.lbl_count.setStyleSheet("color:#5b6b7c; padding:4px 0;")
        lay.addWidget(self.lbl_count)

        # кнопки
        buttons = QHBoxLayout()
        self.btn_open = QPushButton("Открыть результат")
        self.btn_open.clicked.connect(self.open_selected)
        self.btn_delete = QPushButton("Удалить запись")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_delete_all = QPushButton("Удалить все")
        self.btn_delete_all.clicked.connect(self.delete_all)
        buttons.addWidget(self.btn_open)
        buttons.addWidget(self.btn_delete)
        buttons.addWidget(self.btn_delete_all)
        buttons.addStretch(1)
        lay.addLayout(buttons)

        self.refresh()

    def refresh(self) -> None:
        _label, sort, order = _SORT_OPTIONS[self.cmb_order.currentIndex()]
        self._rows = db.search_history(limit=500, order=order, sort=sort)

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._rows))
        for i, r in enumerate(self._rows):
            conf = r.get("confidence", 0)
            conf_text = f"{round(conf)}" if conf is not None else "—"
            values = [
                (str(r.get("result_id", "")), None),
                (str(r.get("patient_code") or "—"), None),
                (f"{r.get('volume_ml', 0):,.1f}" if r.get("volume_ml") is not None else "—",
                 r.get("volume_ml")),
                (conf_text, conf if conf is not None else None),
                (f"{r.get('voxel_count', 0):,}" if r.get("voxel_count") is not None else "—",
                 r.get("voxel_count")),
                (f"{r.get('processing_time', 0):.2f}" if r.get("processing_time") is not None else "—",
                 r.get("processing_time")),
                (str(r.get("model_version") or "—"), None),
                (str(r.get("created_at") or "—"), None),
            ]
            for j, (text, num) in enumerate(values):
                item = QTableWidgetItem(text)
                if num is not None:
                    item.setData(Qt.EditRole, float(num))
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True)

        self.lbl_count.setText(f"Всего записей: {len(self._rows)}")

    def _selected_id(self) -> Optional[int]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return int(self._rows[row]["result_id"])

    def open_selected(self) -> None:
        rid = self._selected_id()
        if rid is None:
            QMessageBox.information(self, "Не выбрано", "Выберите запись в таблице.")
            return
        if self.on_open:
            self.on_open(rid)
            self.accept()

    def delete_selected(self) -> None:
        rid = self._selected_id()
        if rid is None:
            QMessageBox.information(self, "Не выбрано", "Выберите запись в таблице.")
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Удаление")
        msg.setText(f"Удалить запись #{rid} из истории?")
        btn_yes = msg.addButton("Да", QMessageBox.YesRole)
        btn_no = msg.addButton("Нет", QMessageBox.NoRole)
        msg.setDefaultButton(btn_no)
        msg.exec_()
        if msg.clickedButton() == btn_yes:
            db.delete_result(rid)
            # Сбрасываем ID если это была последняя запись
            remaining = db.search_history(limit=1)
            if not remaining:
                db.delete_all_results()
            self.refresh()

    def delete_all(self) -> None:
        if not self._rows:
            QMessageBox.information(self, "Удаление", "История пуста.")
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("Удаление всех записей")
        msg.setText("Удалить все записи из истории?")
        btn_yes = msg.addButton("Да", QMessageBox.YesRole)
        btn_no = msg.addButton("Нет", QMessageBox.NoRole)
        msg.setDefaultButton(btn_no)
        msg.exec_()
        if msg.clickedButton() == btn_yes:
            db.delete_all_results()
            self.refresh()
