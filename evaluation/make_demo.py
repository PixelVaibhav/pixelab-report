"""Build the sample report behind the 'Try a sample report' button (no AI call needed)."""
import json
import os

import fitz

from evaluation.make_samples import make_pdf, status
from rag import KB, norm

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo")
ROWS = [("Hemoglobin", 11.2, "g/dL", 13.0, 17.0), ("RBC", 4.9, "mill/cumm", 4.5, 5.5),
        ("WBC", 12.4, "10^3/uL", 4.0, 11.0), ("Platelets", 250, "10^3/uL", 150, 410),
        ("MCV", 78.0, "fL", 83, 101), ("Fasting Glucose", 112, "mg/dL", 70, 99),
        ("ALT (SGPT)", 62, "U/L", 0, 45), ("Total Cholesterol", 228, "mg/dL", 0, 200),
        ("Vitamin D", 18, "ng/mL", 30, 100)]


def build():
    os.makedirs(OUT, exist_ok=True)
    pdf = os.path.join(OUT, "sample.pdf")
    make_pdf(pdf, "Sample Health Check (demo)", ROWS).close()
    page = fitz.open(pdf)[0]
    W, H = page.rect.width, page.rect.height
    words, kb = page.get_text("words"), KB()
    items, expl, src = [], {}, {}
    for i, (name, v, unit, lo, hi) in enumerate(ROWS):
        r = page.search_for(name)[0]
        row = [w for w in words if r.y0 - 2 <= (w[1] + w[3]) / 2 <= r.y1 + 2]
        box = [min(w[1] for w in row) / H * 1000, min(w[0] for w in row) / W * 1000,
               max(w[3] for w in row) / H * 1000, max(w[2] for w in row) / W * 1000]
        st, d = status(v, lo, hi), kb.alias[norm(name)]
        items.append({"id": i, "page": 1, "kind": "lab", "name": name, "value_text": str(v), "value_number": float(v),
                      "unit": unit, "ref_text": f"{lo}-{hi}", "ref_min": lo, "ref_max": hi, "flag": "unclear",
                      "status": st, "box": box, "located": True, "verified": True})
        text = f"This test measures {d['measures']} "
        text += ("This result is within the printed range." if st == "NORMAL" else
                 f"This result is {st.lower()} compared with the printed range. A {st.lower()} result can be linked to "
                 f"{d['low'] if st == 'LOW' else d['high']} Ask your doctor what it means for you.")
        expl[str(i)] = text
        src[str(i)] = {"kb_id": d["id"], "title": d["name"], "score": 1.0, "method": "alias"}
    good = [it["name"] for it in items if it["status"] == "NORMAL"]
    bad = [it["name"] for it in items if it["status"] != "NORMAL"]
    explain = {
        "good_summary": f"In this sample report, {len(good)} results are within the printed range: {', '.join(good)}.",
        "concern_summary": f"{len(bad)} results are outside the printed range: {', '.join(bad)}. This is a made-up "
                           "sample, but on a real report you should talk to a doctor about any result outside its range.",
        "doctor_points": ["Which of the results outside the range matter most for me?", "Do any of these need a repeat test?",
                          "Could my diet, medicines or a recent illness explain these results?", "When should I test again?"],
        "explanations": expl,
        "spoken_english": "This is a sample report. Some results are inside the printed range and some are outside it. "
                          "For any result outside the range, please talk to a doctor.",
        "spoken_hindi": "यह एक नमूना रिपोर्ट है। इसमें कुछ नतीजे छपी हुई सीमा के अंदर हैं और कुछ बाहर हैं। "
                        "जो नतीजे सीमा से बाहर हैं, उनके बारे में कृपया डॉक्टर से बात करें।"}
    with open(os.path.join(OUT, "demo.json"), "w", encoding="utf8") as f:
        json.dump({"name": "Sample health check (demo)", "doc_type": "lab", "modality": None, "body_part": None,
                   "items": items, "explain": explain, "sources": src}, f, ensure_ascii=False, indent=1)
    print("demo written to", OUT)


if __name__ == "__main__":
    build()
