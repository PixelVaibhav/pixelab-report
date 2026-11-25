import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from evaluation.scoring import health, score, summarize
from rag import KB
from schemas import parse_page


def test_status_from_numbers():
    f = lambda **k: main.get_status({"kind": "lab", **k})
    assert f(value_number=9, ref_min=12, ref_max=17) == "LOW"
    assert f(value_number=20, ref_min=12, ref_max=17) == "HIGH"
    assert f(value_number=15, ref_min=12, ref_max=17) == "NORMAL"
    assert f(flag="abnormal") == "ABNORMAL"
    assert f(flag="???") == "UNCLEAR"
    assert main.get_status({"kind": "medication"}) == "INFO"


def test_clean_box_limits():
    assert main.clean_box([100, 50, 120, 900]) == [100, 50, 120, 900]
    assert main.clean_box([100, 50, 300, 900]) is None          # too tall for a lab row
    assert main.clean_box([100, 50, 400, 900], "finding") is None
    assert main.clean_box([100, 50, 220, 900], "finding") is not None
    assert main.clean_box("junk") is None


def test_pick_is_evenly_spaced():
    assert main.pick(list(range(100)), 3) == [0, 50, 99]
    assert main.pick([1, 2], 5) == [1, 2]


def test_parse_page_validates_and_drops_bad_items():
    page, bad = parse_page({"page_type": "lab_table", "title": "null", "items": [
        {"name": "Hb", "value_number": "15.1", "flag": "weird", "box_2d": [1, 2, 3]},
        {"value_number": 3}]})
    assert bad == 1 and len(page.items) == 1
    it = page.items[0]
    assert it.value_number == 15.1 and it.flag == "unclear" and it.box_2d is None and page.title is None
    assert parse_page([{"name": "TSH"}])[0].items[0].kind == "lab"
    with pytest.raises(ValueError):
        parse_page("not json")


def test_collect_builds_items_with_status():
    res = [{"title": "CBC", "items": [{"name": "Hemoglobin", "kind": "lab", "value_number": 9.0,
                                      "ref_min": 12, "ref_max": 17, "flag": "unclear", "box_2d": None}]}]
    items, scans, meta, failed = main.collect(res, [(b"", b"", None)])
    assert items[0]["status"] == "LOW" and meta["title"] == "CBC" and failed == 0
    assert main.collect([None], [(b"", b"", None)])[3] == 1


def test_rag_alias_retrieval_without_api():
    hits = KB().retrieve_many(None, ["SGPT", "Hb", "Some Unknown Test"])
    assert hits[0]["kb_id"] == "alt" and hits[1]["kb_id"] == "hemoglobin" and hits[2] is None


def test_scoring():
    truth = [{"name": "Hemoglobin", "value_number": 11.2, "status": "LOW"},
             {"name": "ALT (SGPT)", "value_number": 62, "status": "HIGH"}]
    pred = [{"kind": "lab", "name": "Hb", "value_number": 11.2, "status": "LOW"},
            {"kind": "lab", "name": "SGPT", "value_number": 26, "status": "NORMAL"}]
    r = score(pred, truth)
    assert (r["matched"], r["value_ok"], r["status_ok"]) == (2, 1, 1)
    assert summarize([r])["recall"] == 1.0


def test_rate_limit():
    class Req:
        headers = {}
        client = type("C", (), {"host": "9.9.9.9"})()
    main.HITS.clear()
    for _ in range(main.RATE_N):
        main.limit(Req())
    with pytest.raises(HTTPException) as e:
        main.limit(Req())
    assert e.value.status_code == 429


def test_api_metrics_and_missing_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    main.HITS.clear()
    c = TestClient(main.app)
    assert "tokens_in" in c.get("/api/metrics").json()
    assert c.post("/api/analyze", files=[("files", ("a.pdf", b"x"))]).status_code == 503


def test_health_flags_failed_runs():
    assert health(0, 4) == "ok" and health(1, 4) == "degraded" and health(4, 4) == "unavailable"


def test_client_accepts_timeout(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "dummy")
    assert main.client() is not None


def test_retryable_only_for_temporary_errors():
    class E(Exception):
        def __init__(self, code):
            self.code = code
    assert main.retryable(E(503)) and main.retryable(E(429)) and main.retryable(Exception("timeout"))
    assert not main.retryable(E(404)) and not main.retryable(E(401))


def test_verify_checks_value_against_pdf_text():
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((40, 80), "Hemoglobin 15.1 g/dL  reference 13.0 - 17.0 normal range")
    page = doc[0]
    assert main.verify(page, {"kind": "lab", "value_text": "15.1"}) is True
    assert main.verify(page, {"kind": "lab", "value_text": "99.9"}) is False
    assert main.verify(None, {"kind": "lab", "value_text": "15.1"}) is None
    assert main.verify(fitz.open().new_page(), {"kind": "lab", "value_text": "1"}) is None


def test_healthz_and_demo_need_no_api_key(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    c = TestClient(main.app)
    assert c.get("/healthz").json() == {"ok": True}
    d = c.get("/api/demo").json()
    assert d["demo"] and len(d["items"]) >= 5 and all(i["located"] for i in d["items"])
    assert c.get(f"/api/report/{d['id']}/page/1.png").status_code == 200
    assert c.get(f"/api/report/{d['id']}/explain?language=Hindi").status_code == 400


def test_busy_server_returns_503(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    for _ in range(main.MAX_JOBS):
        main.JOBS.acquire()
    try:
        r = TestClient(main.app).post("/api/analyze", files=[("files", ("a.pdf", b"x"))])
        assert r.status_code == 503
    finally:
        for _ in range(main.MAX_JOBS):
            main.JOBS.release()
