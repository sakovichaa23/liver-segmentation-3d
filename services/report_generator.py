# Формирование PDF-отчёта (A4, одна страница)
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

BLUE = (0.11, 0.31, 0.54)
GREY = (0.42, 0.48, 0.55)


def _register_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path in [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("MainFont", path))
                return "MainFont"
            except Exception:  
                continue
    return "Helvetica"


def make_report(path: str | Path, patient_code: str, file_name: str,
                slice_rgb: np.ndarray, params: dict,
                volume_ml: float,
                model_version: str = "", processing_time: float = 0.0) -> Path:
    # Создать PDF-отчёт
    import cv2
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    font = _register_font()

    tmp_img = path.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp_img), cv2.cvtColor(slice_rgb, cv2.COLOR_RGB2BGR))

    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=14 * mm, bottomMargin=14 * mm,
        title="Отчёт по сегментации печени",
    )

    title_st = ParagraphStyle("t", fontName=font, fontSize=18, textColor=colors.HexColor("#1d4e89"),
                              spaceAfter=2, leading=22)
    sub_st = ParagraphStyle("s", fontName=font, fontSize=9.5, textColor=colors.HexColor("#7a8794"),
                            spaceAfter=10, leading=12)
    h_st = ParagraphStyle("h", fontName=font, fontSize=11.5, textColor=colors.HexColor("#1d4e89"),
                          spaceBefore=10, spaceAfter=5, leading=14)
    body_st = ParagraphStyle("b", fontName=font, fontSize=10, textColor=colors.HexColor("#22303f"),
                             leading=15)
    big_st = ParagraphStyle("big", fontName=font, fontSize=20, textColor=colors.HexColor("#1d4e89"),
                            alignment=1, leading=24)
    cap_st = ParagraphStyle("cap", fontName=font, fontSize=9, textColor=colors.HexColor("#7a8794"),
                            alignment=1, leading=12)
    foot_st = ParagraphStyle("f", fontName=font, fontSize=8.5, textColor=colors.HexColor("#98a4b0"),
                             leading=11)

    story = []
    story.append(Paragraph("Отчёт по сегментации печени", title_st))
    story.append(Paragraph(
        "Автоматическая 3D-сегментация и оценка объёма по данным компьютерной томографии", sub_st))

    header_tbl = Table(
        [[Paragraph(f"<b>Код пациента:</b> {patient_code}", body_st),
          Paragraph(f"<b>Дата:</b> {params.get('date', '')}", body_st)],
         [Paragraph(f"<b>Файл:</b> {file_name}", body_st),
          Paragraph(f"<b>Срезов:</b> {params.get('n_slices', '—')}", body_st)]],
        colWidths=[88 * mm, 90 * mm],
    )
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6fa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde3ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde3ea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Объём печени", cap_st))
    story.append(Paragraph(f"{volume_ml:,.1f} мл", big_st))
    story.append(Paragraph(f"{volume_ml / 1000:.2f} л", cap_st))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Срез КТ с наложением маски", h_st))
    img_w = 178 * mm
    img_h = img_w * slice_rgb.shape[0] / slice_rgb.shape[1]
    max_h = 80 * mm
    if img_h > max_h:
        img_h = max_h
        img_w = img_h * slice_rgb.shape[1] / slice_rgb.shape[0]
    img_tbl = Table([[RLImage(str(tmp_img), width=img_w, height=img_h)]], colWidths=[178 * mm])
    img_tbl.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde3ea")),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Технические параметры", h_st))
    spacing = params.get("spacing", ("—", "—", "—"))
    rows = [
        ["Размер среза", str(params.get("image_shape", "—"))],
        ["Шаг по Z", f"{spacing[0]} мм"],
        ["Шаг в плоскости", f"{spacing[2]:.2f} × {spacing[1]:.2f} мм"],
        ["Объём вокселя", f"{params.get('voxel_mm3', 0):.4f} мм³"],
        ["Вокселей печени", f"{params.get('liver_voxels', 0):,}"],
        ["Версия модели", str(model_version or "—")],
        ["Время обработки", f"{processing_time:.2f} с"],
    ]

    data = [[Paragraph(f"<b>{k}</b>", body_st), Paragraph(str(v), body_st)] for k, v in rows]
    tbl = Table(data, colWidths=[70 * mm, 108 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f6fa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde3ea")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dde3ea")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 16))

    story.append(Paragraph(
        "Отчёт является результатом работы исследовательского программного инструмента, "
        "а не самостоятельным медицинским заключением. Все результаты должны проверяться врачом-рентгенологом.", foot_st))

    doc.build(story)
    tmp_img.unlink(missing_ok=True)
    log.info("PDF-отчёт сохранён: %s", path)
    return path
