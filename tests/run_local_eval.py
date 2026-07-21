"""Local evaluation harness (no Docker required).

Runs the router directly over tests/sample/tasks.json and applies lightweight,
deterministic checks derived from the public validation set's expected answers.
It reports per-task pass/fail, the route taken, and the total Fireworks tokens
used — the number that determines Track 1 ranking.

This is NOT the official LLM-judge; it is a fast structural sanity check. A real
answer can fail a heuristic and still pass the judge (and vice-versa), so read
the printed answers too.

Run from the project root:
    python -m tests.run_local_eval
    # force pure-local (zero Fireworks tokens):
    FIREWORKS_MODE=never python -m tests.run_local_eval
"""
from __future__ import annotations

import json
import os
import re
import sys

# Make 'src' importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import router  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "sample", "tasks.json")


def _sentences(text: str) -> int:
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return len(parts)


def _bullets(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [l for l in lines if re.match(r"^([-*•]|\d+[.)])\s+", l)]


def check_T01(a):   # RGB
    lo = a.lower()
    return all(c in lo for c in ("red", "green", "blue")) and "additive" in lo
def check_T01b(a):
    lo = a.lower()
    return "neural" in lo and ("subset" in lo or "type of" in lo or "branch" in lo)
def check_T01c(a):
    lo = a.lower()
    return "volatile" in lo and "ram" in lo and "rom" in lo
def check_T02(a):
    return re.search(r"1[,\.]?672", a) is not None
def check_T02b(a):
    has_cups = re.search(r"1\.8[758]", a) is not None
    has_cost = re.search(r"\$?4\.50", a) is not None
    return has_cups and has_cost
def _mixed_reason(a):
    lo = a.lower()
    label_ok = any(w in lo for w in ("mixed", "neutral", "positive")) and "negative" not in lo.split("\n")[0][:40]
    both = ("but" in lo or "however" in lo or ("," in a)) and (
        any(n in lo for n in ("late", "damaged", "dented", "missing")) and
        any(p in lo for p in ("worked", "flawless", "resolved", "support", "setup", "set up", "fast")))
    return label_ok and both
def check_T03(a):  return _mixed_reason(a)
def check_T03b(a): return _mixed_reason(a)
def check_T04(a):  return _sentences(a) == 2
def check_T04b(a):
    b = _bullets(a)
    return len(b) == 3 and all(len(re.sub(r"^([-*•]|\d+[.)])\s+", "", x).split()) <= 15 for x in b)
def check_T05(a):
    lo = a.lower()
    needed = ["sundar pichai", "google", "zurich", "eth zurich", "march 15"]
    labels = ["person", "organization", "location", "date"]
    return all(n in lo for n in needed) and sum(l in lo for l in labels) >= 3


CHECKS = {
    "T01": check_T01, "T01b": check_T01b, "T01c": check_T01c,
    "T02": check_T02, "T02b": check_T02b,
    "T03": check_T03, "T03b": check_T03b,
    "T04": check_T04, "T04b": check_T04b, "T05": check_T05,
}


def main() -> int:
    with open(TASKS, encoding="utf-8") as f:
        tasks = json.load(f)

    total_fw = 0
    passed = 0
    print("=" * 72)
    for t in tasks:
        tid, prompt = t["task_id"], t["prompt"]
        d = router.route(prompt)
        total_fw += d.fireworks_tokens
        ok = CHECKS.get(tid, lambda a: bool(a.strip()))(d.answer or "")
        passed += int(ok)
        print(f"[{ 'PASS' if ok else 'FAIL' }] {tid:5s} route={d.route:9s} "
              f"cat={d.category:24s} fw_tokens={d.fireworks_tokens}")
        print("   " + (d.answer or "").replace("\n", "\n   ")[:600])
        print("-" * 72)

    print(f"\nAccuracy (heuristic): {passed}/{len(tasks)}")
    print(f"TOTAL FIREWORKS TOKENS (lower = better rank): {total_fw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
