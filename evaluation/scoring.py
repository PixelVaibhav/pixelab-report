"""Scoring for the extraction pipeline against hand-labelled ground truth."""
from rag import KB, norm

_kb = KB()


def canon(name):
    d = _kb.alias.get(norm(name))
    return d["id"] if d else norm(name)


def score(pred, truth):
    """pred and truth are lists of dicts with name, value_number and status."""
    p = {canon(i["name"]): i for i in pred if i.get("kind", "lab") == "lab"}
    r = {"truth": len(truth), "pred": len(p), "matched": 0, "value_ok": 0, "status_ok": 0}
    for t in truth:
        m = p.get(canon(t["name"]))
        if not m:
            continue
        v = m.get("value_number")
        r["matched"] += 1
        r["value_ok"] += v is not None and abs(v - t["value_number"]) < 0.01
        r["status_ok"] += m.get("status") == t["status"]
    return r


def health(failed, total):
    """Tell a real score apart from a run where the model failed to answer."""
    return "unavailable" if total and failed == total else "degraded" if failed else "ok"


def summarize(rows):
    t = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}
    div = lambda a, b: round(a / b, 3) if b else 0.0
    return {"recall": div(t.get("matched", 0), t.get("truth", 0)),
            "precision": div(t.get("matched", 0), t.get("pred", 0)),
            "value_accuracy": div(t.get("value_ok", 0), t.get("matched", 0)),
            "status_accuracy": div(t.get("status_ok", 0), t.get("matched", 0))}
