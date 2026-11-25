"""Run the real pipeline on labelled cases; report accuracy, latency, tokens and cost.

    python -m evaluation.run_eval --markdown
    python -m evaluation.run_eval --cases evaluation/real --models modelA,modelB --markdown

A case is FILE.pdf/.jpg/.png plus FILE.json: [{"name": "...", "value_number": 1.0, "status": "LOW|NORMAL|HIGH"}]
Set PRICE_IN_PER_M and PRICE_OUT_PER_M (USD per million tokens, from Google's pricing page) to get cost.
"""
import argparse
import glob
import json
import os
import statistics
import time

import main
from evaluation.scoring import health, score, summarize
from metrics import PRICE_IN, PRICE_OUT, Usage

HERE = os.path.dirname(__file__)


def run_case(c, path):
    with open(path, "rb") as f:
        data = f.read()
    pages, keep = main.load_pages([(os.path.basename(path), data)])
    u, t0 = Usage(), time.time()
    results = [main.extract_page(c, p[1], u) for p in pages]
    items, _, _, failed = main.collect(results, pages)
    n = len(pages)
    for d in keep:
        d.close()
    return items, u, time.time() - t0, failed, n


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "cases"))
    ap.add_argument("--models", default=main.MODEL)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--markdown", action="store_true", help="print a table to paste into the README")
    a = ap.parse_args()
    c = main.client()
    files = sorted(f for ext in ("pdf", "jpg", "jpeg", "png") for f in glob.glob(os.path.join(a.cases, f"*.{ext}")))
    files = [f for f in files if os.path.exists(os.path.splitext(f)[0] + ".json")]
    report = {}
    for model in a.models.split(","):
        main.MODEL = model.strip()
        rows, times, tin, tout, bad, failed, total, unver = [], [], 0, 0, 0, 0, 0, 0
        print(f"\n== {main.MODEL} ==")
        for f in files:
            truth = json.load(open(os.path.splitext(f)[0] + ".json", encoding="utf8"))
            items, u, s, fp, n = run_case(c, f)
            rows.append(score(items, truth))
            times.append(s); tin += u.tokens_in; tout += u.tokens_out; bad += u.invalid
            failed += fp; total += n
            unver += sum(i.get("verified") is False for i in items)
            r = rows[-1]
            print(f"  {os.path.basename(f)}: {r['matched']}/{r['truth']} rows found, {fp}/{n} pages failed, {s:.0f}s")
        k, status = max(len(files), 1), health(failed, total)
        res = {"status": status, "failed_pages": f"{failed}/{total}", **summarize(rows), "cases": len(files),
               "median_seconds": round(statistics.median(times), 1) if times else 0,
               "max_seconds": round(max(times), 1) if times else 0,
               "tokens_in_per_report": tin // k, "tokens_out_per_report": tout // k,
               "invalid_items": bad, "unverified_values": unver}
        if PRICE_IN or PRICE_OUT:
            res["cost_per_report_usd"] = round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6 / k, 5)
        report[main.MODEL] = res
        print(f"{main.MODEL}: {res}")
        if status != "ok":
            print(f"  WARNING: {status.upper()} run. {failed} of {total} pages got no answer, so recall is not a reading-accuracy score.")
    with open(a.out, "w", encoding="utf8") as f:
        json.dump(report, f, indent=2)
    print("saved", a.out)
    if a.markdown:
        print("\n| Model | Reports | Recall | Value accuracy | Status accuracy | Median s | Cost per report |\n|---|---|---|---|---|---|---|")
        for m, r in report.items():
            print(f"| {m} | {r['cases']} | {r['recall']:.2f} | {r['value_accuracy']:.2f} | {r['status_accuracy']:.2f} | "
                  f"{r['median_seconds']} | {('$' + str(r['cost_per_report_usd'])) if 'cost_per_report_usd' in r else 'set prices'} |")


if __name__ == "__main__":
    cli()
