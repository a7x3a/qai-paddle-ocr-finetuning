"""Generate realistic full-page Kurdish document samples (JPG and multi-page PDF) for the OCR Studio."""

from __future__ import annotations

import os
from pathlib import Path
import fitz
import numpy as np
import cv2

OUTPUT_DIR = Path("data/kurdish_rec").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JPG_PATH = OUTPUT_DIR / "sample_page.jpg"
PDF_PATH = OUTPUT_DIR / "sample_document.pdf"

# Find suitable system font with Arabic/Kurdish support
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]
selected_font = None
for f in FONT_CANDIDATES:
    if os.path.exists(f):
        selected_font = f
        break

if not selected_font:
    raise FileNotFoundError("No system Kurdish/Arabic font found.")

def build_pdf_document():
    doc = fitz.open()

    # --- Page 1: Structured Article with Title, Two Columns, and Table ---
    p1 = doc.new_page(width=595, height=842)  # A4 size
    p1.insert_font(fontname="kurdish_font", fontfile=selected_font)

    # Decorative header rule
    p1.draw_line(fitz.Point(50, 45), fitz.Point(545, 45), color=(0.1, 0.4, 0.6), width=1.5)

    # Document Header / Title
    p1.insert_text(fitz.Point(120, 85), "کوردستان و مێژووی دێرینی ناوچەکە", fontname="kurdish_font", fontsize=20, color=(0.1, 0.15, 0.25))
    p1.insert_text(fitz.Point(160, 115), "بەشی یەکەم: بنەما سەرەکییەکان و سەرەتای شارستانییەت", fontname="kurdish_font", fontsize=13, color=(0.3, 0.35, 0.45))

    p1.draw_line(fitz.Point(50, 135), fitz.Point(545, 135), color=(0.85, 0.88, 0.92), width=1.0)

    # Section 1 Header
    p1.insert_text(fitz.Point(340, 170), "١. پێشەکی و چوارچێوەی گشتی", fontname="kurdish_font", fontsize=14, color=(0.08, 0.45, 0.65))

    # Right Column (Kurdish text reads right-to-left)
    col1_lines = [
        "ناوچەی کوردستان لە ڕووی شوێنگەی جوگرافییەوە یەکێکە",
        "لە دێرینترین شوێنەوارە مرۆییەکانی ڕۆژهەڵاتی ناوەڕاست.",
        "بەپێی پشکنینە شوێنەوارییەکان لە ئەشکەوتی شانەدەر",
        "مرۆڤی نیاندەرتاڵ پێش زیاتر لە شەست هەزار ساڵ لێرە ژیاوە.",
        "ئەمەش دەریدەخات کە ئەم خاکە مێژوویەکی زۆر کۆنی هەیە.",
        "شارستانییەتی گوندنشینی سەرەتایی لە جەرمۆ دەستی پێکرد.",
    ]
    y_pos = 205
    for line in col1_lines:
        p1.insert_text(fitz.Point(295, y_pos), line, fontname="kurdish_font", fontsize=11, color=(0.15, 0.15, 0.18))
        y_pos += 26

    # Left Column
    col2_lines = [
        "پەرەسەندنی کشتوکاڵ و ئاژەڵداری لەم دەڤەرەدا بنەمای",
        "سەرەکی بووە بۆ دروستبوونی یەکەمین شار و کۆمەڵگەکان.",
        "سەرچاوە مێژووییەکان ئاماژە بە چەندین دەسەڵاتداری",
        "وەک لۆلۆبی و گوتی و میتانییەکان دەکەن لە ناوچەکەدا.",
        "هەروەها پاشماوەی نووسینە مێخییەکان لەسەر بەردەکان",
        "شاھیدی لەسەر کلتوور و زمانی دەوڵەمەندی گەلانی دێرینن.",
    ]
    y_pos = 205
    for line in col2_lines:
        p1.insert_text(fitz.Point(50, y_pos), line, fontname="kurdish_font", fontsize=11, color=(0.15, 0.15, 0.18))
        y_pos += 26

    p1.draw_line(fitz.Point(50, 385), fitz.Point(545, 385), color=(0.85, 0.88, 0.92), width=1.0)

    # Section 2: Table Section
    p1.insert_text(fitz.Point(350, 420), "٢. خشتەی گرنگترین ڕووداوە مێژووییەکان", fontname="kurdish_font", fontsize=13, color=(0.08, 0.45, 0.65))

    # Table Header box
    p1.draw_rect(fitz.Rect(50, 440, 545, 470), color=(0.85, 0.9, 0.95), fill=(0.92, 0.96, 0.99), width=1)
    p1.insert_text(fitz.Point(460, 460), "ساڵ و بەروار", fontname="kurdish_font", fontsize=11, color=(0.1, 0.2, 0.3))
    p1.insert_text(fitz.Point(260, 460), "ڕووداو و دەسکەوت", fontname="kurdish_font", fontsize=11, color=(0.1, 0.2, 0.3))
    p1.insert_text(fitz.Point(85, 460), "شوێن و دەڤەر", fontname="kurdish_font", fontsize=11, color=(0.1, 0.2, 0.3))

    table_data = [
        ("١٩٢٠ زاینی", "پەیماننامەی سیڤەر و ناساندنی مافەکان", "پاریس - فەرەنسا"),
        ("١٩٤٦ زاینی", "دامەزراندنی کۆماری دیموکراتی کوردستان", "مەهاباد"),
        ("١٩٩١ زاینی", "ڕاپەڕینی گەلی کوردستان دژی ستەمکاری", "هەولێر و سلێمانی"),
    ]
    t_y = 495
    for date, event, location in table_data:
        p1.draw_rect(fitz.Rect(50, t_y - 20, 545, t_y + 10), color=(0.9, 0.92, 0.94), fill=(0.98, 0.99, 1.0) if (t_y % 50 == 0) else None, width=0.5)
        p1.insert_text(fitz.Point(460, t_y), date, fontname="kurdish_font", fontsize=10.5, color=(0.2, 0.2, 0.2))
        p1.insert_text(fitz.Point(230, t_y), event, fontname="kurdish_font", fontsize=10.5, color=(0.2, 0.2, 0.2))
        p1.insert_text(fitz.Point(80, t_y), location, fontname="kurdish_font", fontsize=10.5, color=(0.2, 0.2, 0.2))
        t_y += 32

    # Concluding paragraph
    p1.insert_text(fitz.Point(365, 630), "٣. ئەنجامگیری و پەیامی سەرەکی", fontname="kurdish_font", fontsize=13, color=(0.08, 0.45, 0.65))
    concl_lines = [
        "پاراستنی ئەم کەلەپوورە دەوڵەمەندە و دۆکیۆمێنتکردنی دەقە کۆنەکان",
        "ئەرکێکی هاوبەشی هەموو ناوەندە ئەکادیمی و توێژینەوەییەکانە.",
        "بەکارهێنانی تەکنەلۆژیای نوێی خوێندنەوەی دەقەکان خزمەتێکی گەورەیە.",
    ]
    c_y = 660
    for line in concl_lines:
        p1.insert_text(fitz.Point(120, c_y), line, fontname="kurdish_font", fontsize=11, color=(0.15, 0.15, 0.18))
        c_y += 26

    # Footer
    p1.draw_line(fitz.Point(50, 780), fitz.Point(545, 780), color=(0.85, 0.88, 0.92), width=1.0)
    p1.insert_text(fitz.Point(220, 805), "لاپەڕە ١ لە ٢ - چاپخانەی نیشتمانی کوردستان", fontname="kurdish_font", fontsize=9.5, color=(0.5, 0.55, 0.6))

    # --- Page 2: Second Page of Document ---
    p2 = doc.new_page(width=595, height=842)
    p2.insert_font(fontname="kurdish_font", fontfile=selected_font)

    p2.draw_line(fitz.Point(50, 45), fitz.Point(545, 45), color=(0.1, 0.4, 0.6), width=1.5)
    p2.insert_text(fitz.Point(170, 90), "بەشی دووەم: بەڵگەنامە و سەرچاوە فەرمییەکان", fontname="kurdish_font", fontsize=17, color=(0.1, 0.15, 0.25))

    p2_lines = [
        "لە سەدەی بیستەمدا چەندین دۆکیۆمێنت و سەرچاوەی مێژوویی بڵاوکراونەتەوە",
        "کە تیشک دەخەنە سەر ڕەوتی گۆڕانکارییە کۆمەڵایەتی و ئابوورییەکان.",
        "توێژەران دەتوانن لە ڕێگەی ئەم ئەرشیفانەوە دەستیان بە بەڵگەنامەی گرنگ بگات.",
        "هەروەها پڕۆژەکانی بەدیجیتاڵکردنی دەستنووسەکان هەنگاوێکی گرنگە لەم بوارەدا.",
    ]
    p2_y = 150
    for line in p2_lines:
        p2.insert_text(fitz.Point(90, p2_y), line, fontname="kurdish_font", fontsize=11.5, color=(0.15, 0.15, 0.18))
        p2_y += 30

    p2.draw_line(fitz.Point(50, 780), fitz.Point(545, 780), color=(0.85, 0.88, 0.92), width=1.0)
    p2.insert_text(fitz.Point(220, 805), "لاپەڕە ٢ لە ٢ - چاپخانەی نیشتمانی کوردستان", fontname="kurdish_font", fontsize=9.5, color=(0.5, 0.55, 0.6))

    # Save PDF
    doc.save(str(PDF_PATH))
    print(f"Generated PDF: {PDF_PATH} ({len(doc)} pages)")

    # Render Page 1 to high-resolution JPEG (300 DPI) for sample_page.jpg
    page1 = doc.load_page(0)
    zoom = 300 / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page1.get_pixmap(matrix=matrix, alpha=False)
    img_bgr = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(JPG_PATH), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Generated High-Res JPG: {JPG_PATH} ({pix.width}x{pix.height})")

    doc.close()

if __name__ == "__main__":
    build_pdf_document()
