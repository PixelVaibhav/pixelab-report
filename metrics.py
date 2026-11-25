"""Token, latency and error counters. Never records report contents."""
import os
import threading

PRICE_IN = float(os.getenv("PRICE_IN_PER_M", "0"))    # USD per million input tokens
PRICE_OUT = float(os.getenv("PRICE_OUT_PER_M", "0"))  # USD per million output tokens


class Usage:
    def __init__(self):
        self._l = threading.Lock()
        self.calls = self.tokens_in = self.tokens_out = self.errors = self.invalid = self.analyses = 0
        self.seconds = 0.0

    def add(self, resp):
        u = getattr(resp, "usage_metadata", None)
        with self._l:
            self.calls += 1
            self.tokens_in += getattr(u, "prompt_token_count", 0) or 0
            self.tokens_out += getattr(u, "candidates_token_count", 0) or 0

    def bump(self, field, n=1):
        with self._l:
            setattr(self, field, getattr(self, field) + n)

    def done(self, secs):
        with self._l:
            self.analyses += 1
            self.seconds += secs

    def dict(self):
        d = {"calls": self.calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
             "errors": self.errors, "invalid_items": self.invalid, "analyses": self.analyses,
             "avg_seconds": round(self.seconds / self.analyses, 2) if self.analyses else 0}
        if PRICE_IN or PRICE_OUT:
            d["est_cost_usd"] = round((self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1e6, 4)
        return d


TOTAL = Usage()
