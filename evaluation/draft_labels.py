"""Speed up labelling: the model drafts a labels file, then YOU correct it against the original report.

    python -m evaluation.draft_labels evaluation/real/my_report.pdf

Check EVERY row against the original before evaluating. Unchecked drafts only measure the model against itself.
Keep real reports in evaluation/real/ (git-ignored) and never commit private data.
"""
import json
import os
import sys

import main
from evaluation.run_eval import run_case


def cli():
    path = sys.argv[1]
    out = os.path.splitext(path)[0] + ".json"
    if os.path.exists(out):
        sys.exit(f"{out} already exists; not overwriting.")
    items = run_case(main.client(), path)[0]
    truth = [{"name": i["name"], "value_number": i["value_number"], "status": i["status"]}
             for i in items if i["kind"] == "lab" and i.get("value_number") is not None]
    with open(out, "w", encoding="utf8") as f:
        json.dump(truth, f, indent=1)
    print(f"{len(truth)} rows written to {out}. Now check every row against the original report.")


if __name__ == "__main__":
    cli()
