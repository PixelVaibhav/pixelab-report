import io
import json
import logging
import os
import random
import re
import secrets
import threading
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from gtts import gTTS
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from metrics import TOTAL, Usage
from rag import KB
from schemas import parse_page

try:  # DICOM (MRI / CT / X-ray files)
    import numpy as np
    import pydicom
except ImportError:
    pydicom = None
try:  # iPhone HEIC photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pixelab")
KNOWLEDGE = KB()
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_MB = int(os.getenv("MAX_MB", "40"))
TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "90"))  # per AI call; slow models fail fast and retry
RATE_N = int(os.getenv("RATE_LIMIT", "10"))  # reports per IP per hour
MAX_PAGES, TTL, DPI, SLICES = 50, 1800, 130, 12
WORKERS = int(os.getenv("WORKERS", "4"))
MAX_JOBS = int(os.getenv("MAX_CONCURRENT", "2"))  # simultaneous analyses; protects small servers
JOBS = threading.BoundedSemaphore(MAX_JOBS)
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo")
LANGS = ("English", "Hindi", "Hinglish")
BAD = ("LOW", "HIGH", "ABNORMAL")
INK = {"red": (255, 125, 125), "green": (125, 220, 150), "amber": (255, 215, 110), "blue": (125, 200, 255)}
NOSTORE = {"Cache-Control": "no-store"}
KINDS = ("lab", "finding", "medication")

EXTRACT_PROMPT = """You are reading ONE page or image from a patient's medical document. It may be a lab report, a radiology report (MRI, CT, X-ray, ultrasound), an ECG or other test report, a discharge summary, a prescription, or a raw scan image. It may be a tilted, shadowed or low-quality phone photo: read what is legible and never guess.
Return ONLY a JSON object:
{"page_type": "lab_table|narrative_report|prescription|scan_image|other", "title": "MRI Brain report", "modality": "MRI|CT|X-ray|Ultrasound|ECG|null", "body_part": "Brain|null", "scan_description": null, "items": [{"kind": "lab", "name": "Hemoglobin", "value_text": "15.1", "value_number": 15.1, "unit": "g/dL", "ref_text": "12.5-17.0", "ref_min": 12.5, "ref_max": 17.0, "flag": "normal", "box_2d": [ymin, xmin, ymax, xmax]}]}
Item kinds:
- "lab": one per test result row. box_2d covers the WHOLE printed row, test name to reference range, tight vertically. value_number, ref_min, ref_max are numbers or null (">59" -> ref_min 59; "<5" -> ref_max 5).
- "finding": for narrative reports (radiology, ultrasound, ECG, discharge). One per distinct finding or section, such as Impression, an organ, or a measurement. name is a short label; value_text is the report's own wording shortened to at most 30 words; value_number, unit, ref fields are null unless printed. flag is "abnormal" only if the wording states an abnormality, "normal" if it says normal/unremarkable/no abnormality, otherwise "unclear". box_2d covers those printed lines.
- "medication": one per drug on a prescription. name is the drug, value_text is the dose and frequency as printed, flag "unclear".
- box_2d uses 0-1000 coordinates, y first. Use null if unsure.
- A raw scan image with no report text: items is [] and scan_description is 2-3 plain sentences on what is visibly shown (type of image, view or plane, body region, image quality). Never name a disease or say whether anything is abnormal.
- Use only what is printed. Never invent names, values or ranges. Skip patient details, headers, page numbers. If there is nothing, items is []."""

TYPE_NAME = {"lab": "laboratory report", "report": "medical report (narrative findings)",
             "prescription": "prescription", "scan": "scan image such as MRI, CT or X-ray"}
GOOD = {"lab": "2-3 sentences on what is within the printed range.",
        "report": "2-3 sentences on what the report says in simple words, including anything it calls normal.",
        "prescription": "2-3 sentences on what medicines are listed and what each is generally used for. Do not change or judge doses.",
        "scan": "2-3 sentences on what kind of scan this is and what that kind of scan is generally used to look at, using IMAGE_NOTES. Do not describe findings."}
CONCERN = {"lab": "2-3 sentences on what is outside the range and deserves attention (say clearly if nothing).",
           "report": "2-3 sentences on what the report flags as abnormal or needing follow-up (say clearly if nothing is flagged).",
           "prescription": "2-3 sentences on things to confirm with the doctor or pharmacist, such as side effects to ask about or anything unclear.",
           "scan": "2-3 sentences saying plainly that AI cannot read a scan like a radiologist, cannot find or rule out disease, and that a radiologist or doctor must review the scan."}

app = FastAPI(docs_url=None, redoc_url=None)
STORE, HITS = {}, {}  # in-memory only; nothing is written to disk


def client():
    if not API_KEY:
        raise HTTPException(503, "Server is missing GEMINI_API_KEY.")
    return genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))


def limit(req):
    ip = (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "?")).split(",")[0].strip()
    now = time.time()
    hits = [t for t in HITS.get(ip, []) if now - t < 3600]
    if len(hits) >= RATE_N:
        raise HTTPException(429, "Too many reports from this network. Please try again in an hour.")
    HITS[ip] = hits + [now]
    if len(HITS) > 2000:
        HITS.clear()


def ask(c, contents, usage=None):
    r = c.models.generate_content(
        model=MODEL, contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    for u in (usage, TOTAL):
        if u:
            u.add(r)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    return json.loads(t)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_status(it):
    if it.get("kind") == "medication":
        return "INFO"
    v, lo, hi = num(it.get("value_number")), num(it.get("ref_min")), num(it.get("ref_max"))
    if v is not None and (lo is not None or hi is not None):
        if lo is not None and v < lo:
            return "LOW"
        if hi is not None and v > hi:
            return "HIGH"
        return "NORMAL"
    return {"normal": "NORMAL", "abnormal": "ABNORMAL"}.get(str(it.get("flag")).lower(), "UNCLEAR")


def clean_box(b, kind="lab"):
    """Return a sane box [ymin,xmin,ymax,xmax] (0-1000) or None if not trustworthy."""
    try:
        y1, x1, y2, x2 = [float(v) for v in b]
    except Exception:
        return None
    lab = kind == "lab"
    ok = (4 <= y2 - y1 <= (80 if lab else 250) and x2 - x1 >= (250 if lab else 150)
          and min(y1, x1) >= 0 and max(y2, x2) <= 1000)
    return [y1, x1, y2, x2] if ok else None


def snap(page, box, kind="lab"):
    """Digital PDFs: snap the AI box to the real text line so the stroke sits exactly on the row."""
    words = page.get_text("words")
    W, H = page.rect.width, page.rect.height
    row = [w for w in words if box[0] <= (w[1] + w[3]) / 2 / H * 1000 <= box[2]]
    if not row:
        return box
    snapped = [min(w[1] for w in row) / H * 1000, min(w[0] for w in row) / W * 1000,
               max(w[3] for w in row) / H * 1000, max(w[2] for w in row) / W * 1000]
    return clean_box(snapped, kind) or box


def mark(png, items):
    """Draw translucent highlighter strokes (multiply blend keeps the printed text readable)."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    W, H = img.size
    S = 3
    layer = Image.new("RGB", (W * S, H * S), "white")
    d = ImageDraw.Draw(layer)
    for it in items:
        if not it.get("box"):
            continue
        y1, x1, y2, x2 = it["box"]
        pad = (y2 - y1) / 1000 * H * 0.25
        top, bottom = (y1 / 1000 * H - pad) * S, (y2 / 1000 * H + pad) * S
        left, right = (x1 / 1000 * W - 8) * S, (x2 / 1000 * W + 8) * S
        st = it["status"]
        tone = "red" if st in BAD else "green" if st == "NORMAL" else "blue" if st == "INFO" else "amber"
        d.rounded_rectangle([left, top, right, bottom], radius=min((bottom - top) / 2, 40 * S), fill=INK[tone])
    layer = layer.resize((W, H), Image.LANCZOS)
    return ImageChops.multiply(img, layer)


def jpg(img, q=88):
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=q)
    return b.getvalue()


def enhance(img):
    """Help the AI read shadowed phone photos and faint scans."""
    return ImageOps.autocontrast(img, cutoff=0.5).filter(ImageFilter.UnsharpMask(1.2, 80, 3))


def pick(seq, n):
    if len(seq) <= n:
        return seq
    return [seq[round(i * (len(seq) - 1) / (n - 1))] for i in range(n)]


def add_image(pages, img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((2000, 2000))
    pages.append((jpg(img), jpg(enhance(img)), None))  # (shown to user, sent to AI, pdf page)


def is_dicom(name, data):
    return name.lower().endswith((".dcm", ".dicom")) or data[128:132] == b"DICM"


def dicom_images(data):
    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    a = ds.pixel_array
    rgb = int(ds.get("SamplesPerPixel", 1) or 1) == 3
    frames = [a] if a.ndim == (3 if rgb else 2) else list(a)
    out = []
    for f in pick(frames, SLICES):
        if rgb:
            out.append(Image.fromarray(f.astype("uint8")))
            continue
        f = f.astype("float32") * float(ds.get("RescaleSlope", 1) or 1) + float(ds.get("RescaleIntercept", 0) or 0)
        lo, hi = np.percentile(f, [1, 99.5])
        f = np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            f = 255 - f
        out.append(Image.fromarray(f.astype("uint8")))
    return out


def expand(blobs):
    """Unpack ZIP files (for example a scan CD folder) into their files."""
    out = []
    for name, data in blobs:
        if not name.lower().endswith(".zip"):
            out.append((name, data))
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
            inf = [i for i in z.infolist() if not i.is_dir()][:500]
            if sum(i.file_size for i in inf) > 400 * 1024 * 1024:
                raise ValueError("too big")
            out += [(i.filename, z.read(i)) for i in inf
                    if not os.path.basename(i.filename).startswith((".", "_"))]
        except Exception:
            raise HTTPException(400, f"Could not open ZIP '{name}'.")
    return out


def load_pages(blobs):
    pages, keep, dic, rest = [], [], [], []
    for b in expand(blobs):
        (dic if is_dicom(*b) else rest).append(b)
    if dic:
        if not pydicom:
            raise HTTPException(400, "DICOM files are not supported on this server.")
        meta = []
        for n, d in dic:
            try:
                h = pydicom.dcmread(io.BytesIO(d), stop_before_pixels=True, force=True)
                meta.append((int(h.get("InstanceNumber", 0) or 0), n, d))
            except Exception:
                pass
        imgs = []
        for _, n, d in pick(sorted(meta, key=lambda m: m[0]), SLICES):
            try:
                imgs += dicom_images(d)
            except Exception:
                pass
        if not imgs:
            raise HTTPException(400, "Could not read the scan images in these DICOM files.")
        for im in pick(imgs, SLICES):
            add_image(pages, im)
    for name, data in rest:
        try:
            if name.lower().endswith(".pdf"):
                doc = fitz.open(stream=data, filetype="pdf")
                keep.append(doc)
                if len(pages) + doc.page_count > MAX_PAGES:
                    raise HTTPException(413, f"Please upload at most {MAX_PAGES} pages.")
                for p in doc:
                    scanned = len(p.get_text("words")) < 5  # photo-only PDF: render sharper
                    pix = p.get_pixmap(dpi=170 if scanned else DPI)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    shown = jpg(img)
                    pages.append((shown, jpg(enhance(img)) if scanned else shown, p))
            else:
                add_image(pages, Image.open(io.BytesIO(data)))
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, f"Could not open '{name}'. Use a PDF, photo or DICOM scan.")
    if len(pages) > MAX_PAGES:
        raise HTTPException(413, f"Please upload at most {MAX_PAGES} pages.")
    return pages, keep


def retryable(e):
    """Retry temporary failures (503, 429, timeouts); stop at once on 400, 401, 403 or 404."""
    return getattr(e, "code", None) not in (400, 401, 403, 404)


def extract_page(c, img, usage=None):
    for attempt in range(3):
        try:
            page, bad = parse_page(ask(c, [types.Part.from_bytes(data=img, mime_type="image/jpeg"), EXTRACT_PROMPT], usage))
            if usage:
                usage.bump("invalid", bad)
            return page.model_dump()
        except Exception as e:
            log.warning("page extraction failed (attempt %d): %s", attempt + 1, e)
            for u in (usage, TOTAL):
                if u:
                    u.bump("errors")
            if not retryable(e):
                break
            time.sleep(2 ** attempt + random.random())
    return None


def vs(i):
    return f"{i.get('value_text') or i.get('value_number') or ''} {i.get('unit') or ''}".strip()


def explain(c, items, language, doc_type="lab", scans=(), know=None, usage=None):
    rows = [{"id": i["id"], "test": i["name"], "value": vs(i), "range": i.get("ref_text") or "",
             "status": i["status"]} for i in items]
    kn = [{"id": k, "measures": v["measures"], "if_low": v["low"], "if_high": v["high"]} for k, v in (know or {}).items()]
    prompt = f"""You are helping a patient understand their medical document: a {TYPE_NAME[doc_type]}.
Write text fields in: {language} (Hinglish means Hindi written in English letters). Use very simple words.
Return ONLY a JSON object with keys:
- good_summary: {GOOD[doc_type]}
- concern_summary: {CONCERN[doc_type]}
- doctor_points: list of up to 4 short things to discuss with a doctor.
- explanations: object mapping every id (as a string) in RESULTS to 2 short sentences. Lab test: what it measures and what this result means against its printed range. Finding: what the wording means in plain words and why a doctor may care. Medicine: what it is generally used for. Use an empty object if RESULTS is empty.
- spoken_hindi: warm spoken script of about 120 words in Hindi (Devanagari) covering the same points.
- spoken_english: the same script in simple English.
For any id present in KNOWLEDGE, take what the test measures and the general reasons for low or high results only from those notes.
Strict rules: never diagnose or name a specific disease as the cause, never suggest medicines or doses, never say the person is healthy because values are in range or a report looks normal. You cannot interpret raw scan images like a radiologist. End concern_summary and both scripts with a reminder to talk to a doctor.
RESULTS: {json.dumps(rows, ensure_ascii=False)}
IMAGE_NOTES: {json.dumps(list(scans), ensure_ascii=False)}
KNOWLEDGE: {json.dumps(kn, ensure_ascii=False)}"""
    return ask(c, prompt, usage)


def purge():
    now = time.time()
    for k in [k for k, v in STORE.items() if now - v["ts"] > TTL]:
        del STORE[k]
    while len(STORE) > 15:
        del STORE[min(STORE, key=lambda k: STORE[k]["ts"])]


def get(rid):
    purge()
    r = STORE.get(rid)
    if not r:
        raise HTTPException(404, "This report has expired. Please upload it again.")
    return r


def verify(page, it):
    """Digital PDFs only: is the extracted value really printed on the page? None means cannot tell."""
    if page is None or it.get("kind") != "lab" or not it.get("value_text"):
        return None
    flat = re.sub(r"\s+", "", page.get_text())
    if len(flat) < 20:
        return None
    return re.sub(r"\s+", "", str(it["value_text"])) in flat


def collect(results, pages):
    items, scans, meta, failed = [], [], {}, 0
    for n, (res, (_, _, pdf)) in enumerate(zip(results, pages), start=1):
        if res is None:
            failed += 1
            continue
        for k in ("title", "modality", "body_part"):
            v = res.get(k)
            if v and str(v).lower() not in ("null", "none"):
                meta.setdefault(k, str(v))
        if res.get("scan_description"):
            scans.append(f"Image {n}: {res['scan_description']}")
        for it in res["items"]:
            kind = it.get("kind") if it.get("kind") in KINDS else "lab"
            box = clean_box(it.get("box_2d"), kind)
            if box and pdf is not None:
                box = snap(pdf, box, kind)
            it.update(id=len(items), page=n, kind=kind, status=get_status(it), box=box, located=box is not None,
                      verified=verify(pdf, it))
            it.pop("box_2d", None)
            items.append(it)
    return items, scans, meta, failed


def ground(c, items):
    """RAG step: look up trusted notes for each lab test."""
    labs = [i for i in items if i["kind"] == "lab"]
    hits = KNOWLEDGE.retrieve_many(c, [i["name"] for i in labs])
    return {i["id"]: h for i, h in zip(labs, hits) if h}


def sources(know):
    return {k: {x: v[x] for x in ("kb_id", "title", "score", "method")} for k, v in know.items()}


def finish(usage, t0, n_items):
    secs = time.time() - t0
    usage.done(secs)
    TOTAL.done(secs)
    d = usage.dict()
    d["seconds"] = round(secs, 1)
    log.info("analyzed items=%d seconds=%.1f tokens_in=%d tokens_out=%d invalid=%d errors=%d",
             n_items, secs, d["tokens_in"], d["tokens_out"], d["invalid_items"], d["errors"])
    return d


def _analyze(request, files, language):
    limit(request)
    if language not in LANGS:
        language = "English"
    blobs = [(f.filename or "report", f.file.read()) for f in files]
    if sum(len(b) for _, b in blobs) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Files are too large (limit {MAX_MB} MB).")
    c = client()
    t0, usage = time.time(), Usage()
    pages, keep = load_pages(blobs)
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda p: extract_page(c, p[1], usage), pages))
    items, scans, meta, failed = collect(results, pages)
    for d in keep:
        d.close()
    if not items and not scans:
        raise HTTPException(422, "Nothing readable was found. Try a clearer photo, PDF or scan.")
    cnt = Counter(i["kind"] for i in items)
    doc_type = ("scan" if not items else "lab" if cnt["lab"] >= max(cnt["finding"], cnt["medication"])
                else "report" if cnt["finding"] >= cnt["medication"] else "prescription")
    rid = secrets.token_urlsafe(16)
    try:
        know = ground(c, items)
        exp = explain(c, items, language, doc_type, scans, know, usage)
    except Exception:
        raise HTTPException(502, "The AI service could not write the explanations. Please try again.")
    STORE[rid] = {"ts": time.time(), "pages": [p[0] for p in pages], "items": items, "scans": scans,
                  "doc_type": doc_type, "know": know, "exp": {language: exp}}
    return {"id": rid, "name": meta.get("title") or blobs[0][0], "date": date.today().isoformat(),
            "pages": len(pages), "language": language, "items": items, "explain": exp,
            "doc_type": doc_type, "modality": meta.get("modality"), "body_part": meta.get("body_part"),
            "scans": scans, "unreadable_pages": failed, "sources": sources(know),
            "stats": {**finish(usage, t0, len(items)), "unverified": sum(i.get("verified") is False for i in items)}}


@app.post("/api/analyze")
def analyze(request: Request, files: list[UploadFile] = File(...), language: str = Form("English")):
    if not JOBS.acquire(blocking=False):
        raise HTTPException(503, "The server is busy with other reports. Please try again in a minute.")
    try:
        return _analyze(request, files, language)
    finally:
        JOBS.release()


@app.get("/api/report/{rid}/explain")
def explain_again(rid: str, language: str = "English"):
    r = get(rid)
    if language not in LANGS:
        raise HTTPException(400, "Unknown language.")
    if language not in r["exp"] and r.get("demo"):
        raise HTTPException(400, "The sample report is English only. Upload your own report for other languages.")
    if language not in r["exp"]:
        r["exp"][language] = explain(client(), r["items"], language, r["doc_type"], r["scans"], r["know"])
    return r["exp"][language]


@app.get("/api/report/{rid}/page/{n}.png")
def page_image(rid: str, n: int, marked: int = 1):
    r = get(rid)
    if not 1 <= n <= len(r["pages"]):
        raise HTTPException(404, "No such page.")
    if not marked:
        return Response(r["pages"][n - 1], media_type="image/jpeg", headers=NOSTORE)
    buf = io.BytesIO()
    mark(r["pages"][n - 1], [i for i in r["items"] if i["page"] == n]).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers=NOSTORE)


@app.get("/api/report/{rid}/report.pdf")
def report_pdf(rid: str, marked: int = 1):
    r = get(rid)
    imgs = [mark(b, [i for i in r["items"] if i["page"] == n]) if marked else Image.open(io.BytesIO(b)).convert("RGB")
            for n, b in enumerate(r["pages"], start=1)]
    buf = io.BytesIO()
    imgs[0].save(buf, "PDF", save_all=True, append_images=imgs[1:], resolution=DPI)
    name = "pixelab-marked-report.pdf" if marked else "pixelab-original-report.pdf"
    return Response(buf.getvalue(), media_type="application/pdf",
                    headers={**NOSTORE, "Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/report/{rid}/audio")
def audio(rid: str, lang: str = "en", language: str = "English"):
    r = get(rid)
    script = (r["exp"].get(language) or {}).get("spoken_hindi" if lang == "hi" else "spoken_english")
    if not script:
        raise HTTPException(404, "No audio script available.")
    buf = io.BytesIO()
    try:
        gTTS(text=script, lang="hi" if lang == "hi" else "en").write_to_fp(buf)
    except Exception:
        raise HTTPException(502, "Audio service is unavailable right now.")
    return Response(buf.getvalue(), media_type="audio/mpeg", headers=NOSTORE)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/demo")
def demo():
    """A ready-made sample report: instant, no upload, no AI call."""
    with open(os.path.join(DEMO_DIR, "demo.json"), encoding="utf8") as f:
        d = json.load(f)
    doc = fitz.open(os.path.join(DEMO_DIR, "sample.pdf"))
    pages = []
    for p in doc:
        pix = p.get_pixmap(dpi=DPI)
        pages.append(jpg(Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
    doc.close()
    rid = secrets.token_urlsafe(16)
    STORE[rid] = {"ts": time.time(), "pages": pages, "items": d["items"], "scans": [], "doc_type": "lab",
                  "know": {}, "demo": True, "exp": {"English": d["explain"]}}
    return {**d, "id": rid, "date": date.today().isoformat(), "pages": len(pages), "language": "English",
            "scans": [], "unreadable_pages": 0, "demo": True}


@app.get("/api/metrics")
def metrics():
    return TOTAL.dict()


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="site")
