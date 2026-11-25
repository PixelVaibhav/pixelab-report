"""Small retrieval layer: alias match first, then embedding similarity over a knowledge base."""
import json
import logging
import os
import re

import numpy as np

log = logging.getLogger("pixelab")
KB_PATH = os.path.join(os.path.dirname(__file__), "kb", "tests.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.72"))  # tune this with the evaluation set
SOURCE = "Pixelab educational notes"


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _text(d):
    return f'{d["name"]}. Also called {", ".join(d.get("aliases", []))}. {d["measures"]}'


class KB:
    def __init__(self, path=KB_PATH):
        with open(path, encoding="utf8") as f:
            self.docs = json.load(f)
        self.alias = {norm(a): d for d in self.docs for a in [d["name"], *d.get("aliases", [])]}
        self.vecs = None

    def _hit(self, d, score, method):
        return {"kb_id": d["id"], "title": d["name"], "score": round(score, 3), "method": method,
                "measures": d["measures"], "low": d["low"], "high": d["high"]}

    def _embed(self, client, texts):
        r = client.models.embed_content(model=EMBED_MODEL, contents=texts)
        v = np.array([e.values for e in r.embeddings], dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def retrieve_many(self, client, names):
        """One result per name: a dict, or None when nothing relevant was found."""
        out = [self._hit(self.alias[norm(n)], 1.0, "alias") if norm(n) in self.alias else None for n in names]
        todo = [i for i, o in enumerate(out) if o is None]
        if todo and client is not None:
            try:
                if self.vecs is None:
                    self.vecs = self._embed(client, [_text(d) for d in self.docs])
                sims = self._embed(client, [names[i] for i in todo]) @ self.vecs.T
                for row, i in enumerate(todo):
                    j = int(sims[row].argmax())
                    if sims[row, j] >= MIN_SIM:
                        out[i] = self._hit(self.docs[j], float(sims[row, j]), "embedding")
            except Exception as e:
                log.warning("embedding retrieval failed: %s", e)
        return out
