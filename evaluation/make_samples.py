"""Generate synthetic lab reports with known answers (PDF plus a degraded phone-photo version)."""
import io
import json
import os

import fitz
from PIL import Image, ImageEnhance, ImageFilter

OUT = os.path.join(os.path.dirname(__file__), "cases")
REPORTS = {
    "cbc_basic": ("Complete Blood Count", [
        ("Hemoglobin", 11.2, "g/dL", 13.0, 17.0), ("RBC", 4.9, "mill/cumm", 4.5, 5.5),
        ("WBC", 12.4, "10^3/uL", 4.0, 11.0), ("Platelets", 250, "10^3/uL", 150, 410),
        ("MCV", 78.0, "fL", 83, 101), ("Neutrophils", 62, "%", 40, 80), ("Lymphocytes", 30, "%", 20, 40)]),
    "lipid_liver": ("Lipid and Liver Profile", [
        ("Total Cholesterol", 228, "mg/dL", 0, 200), ("HDL Cholesterol", 38, "mg/dL", 40, 60),
        ("LDL Cholesterol", 152, "mg/dL", 0, 100), ("Triglycerides", 190, "mg/dL", 0, 150),
        ("ALT (SGPT)", 62, "U/L", 0, 45), ("AST (SGOT)", 30, "U/L", 0, 40),
        ("Bilirubin", 0.8, "mg/dL", 0.3, 1.2), ("Albumin", 4.4, "g/dL", 3.5, 5.0)]),
    "sugar_thyroid": ("Diabetes and Thyroid Panel", [
        ("Fasting Glucose", 112, "mg/dL", 70, 99), ("HbA1c", 6.1, "%", 4.0, 5.6),
        ("TSH", 5.8, "uIU/mL", 0.4, 4.2), ("Creatinine", 0.9, "mg/dL", 0.7, 1.3),
        ("Uric Acid", 7.4, "mg/dL", 3.5, 7.2), ("Vitamin D", 18, "ng/mL", 30, 100),
        ("Vitamin B12", 410, "pg/mL", 200, 900)]),
}


def status(v, lo, hi):
    return "LOW" if v < lo else "HIGH" if v > hi else "NORMAL"


def make_pdf(path, title, rows):
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((40, 60), "CITY DIAGNOSTIC LABORATORY", fontsize=16)
    p.insert_text((40, 85), title, fontsize=12)
    p.insert_text((40, 105), "Patient: Sample Person   Age/Sex: 35/M   Date: 01-01-2026", fontsize=9)
    for x, h in zip((40, 220, 300, 400), ("TEST", "RESULT", "UNIT", "REFERENCE RANGE")):
        p.insert_text((x, 140), h, fontsize=9, fontname="hebo")
    for n, (name, v, unit, lo, hi) in enumerate(rows):
        y = 168 + 24 * n
        flag = {"LOW": "L", "HIGH": "H", "NORMAL": ""}[status(v, lo, hi)]
        for x, t in zip((40, 220, 300, 400, 510), (name, str(v), unit, f"{lo} - {hi}", flag)):
            p.insert_text((x, y), t, fontsize=10)
    doc.save(path)
    return doc


def photo(doc, path):
    pix = doc[0].get_pixmap(dpi=110)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = img.rotate(4, expand=True, fillcolor=(110, 105, 100)).filter(ImageFilter.GaussianBlur(0.9))
    shade = Image.linear_gradient("L").resize(img.size).point(lambda v: 150 + v * 105 // 255)
    img = Image.composite(img, Image.new("RGB", img.size, (60, 60, 60)), shade)
    ImageEnhance.Contrast(img).enhance(0.85).save(path, "JPEG", quality=70)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for key, (title, rows) in REPORTS.items():
        truth = [{"name": n, "value_number": v, "status": status(v, lo, hi)} for n, v, _, lo, hi in rows]
        doc = make_pdf(os.path.join(OUT, f"{key}.pdf"), title, rows)
        json.dump(truth, open(os.path.join(OUT, f"{key}.json"), "w"), indent=1)
        if key == "cbc_basic":
            photo(doc, os.path.join(OUT, "cbc_basic_photo.jpg"))
            json.dump(truth, open(os.path.join(OUT, "cbc_basic_photo.json"), "w"), indent=1)
    print("samples written to", OUT)
