"""Per-caller memory: the offline parts (file identity, load, context, guards).

The LLM summarization is exercised on real calls; here we prove the call-path
pieces — which must be fast and failure-proof — behave.

Run: uv run python tests/test_caller_memory.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app import caller_memory

failures = []


def check(cond, note):
    print(("PASS  " if cond else "FAIL  ") + note)
    if not cond:
        failures.append(note)


# --- number -> file identity --------------------------------------------------
p1 = caller_memory._path("09812345678")
p2 = caller_memory._path("+91 98123 45678")
check(p1 is not None and p1 == p2, "same caller, formatted differently -> same file")
check(caller_memory._path("5678") is None, "short/masked caller id -> no identity, no file")
check(caller_memory._path("") is None, "empty caller id -> None")

# --- load guards --------------------------------------------------------------
check(caller_memory.load("00000000000") is None, "unknown number -> None")

# --- roundtrip via a real file ------------------------------------------------
caller_memory.CALLERS_DIR.mkdir(parents=True, exist_ok=True)
test_path = caller_memory._path("07777000111")
test_path.write_text(json.dumps({
    "notes": "Name Manish, from Delhi, wants Wellness Pro, budget 12-15 lakhs, English speaker.",
    "language": "en-IN",
    "calls": 2,
    "last_call_at": time.time(),
    "last_call_at_text": "02 Sep 2026",
}, ensure_ascii=False), encoding="utf-8")
try:
    mem = caller_memory.load("+91 77770 00111")
    check(mem is not None and "Manish" in mem["notes"], "load finds the stored notes")
    check(caller_memory.remembered_language(mem) == "en-IN", "language remembered")
    block = caller_memory.context_block(mem)
    check("RETURNING CALLER" in block and "Manish" in block, "context block carries the notes")
    check("02 Sep 2026" in block, "context block says when the last call was")
    check(len(block) < 2000, "context block stays prompt-cheap")

    # corrupt file -> soft None, never a crash
    test_path.write_text("{not json", encoding="utf-8")
    check(caller_memory.load("07777000111") is None, "corrupt file -> None (soft failure)")
finally:
    test_path.unlink(missing_ok=True)

# --- bad language values never pre-pin ---------------------------------------
check(caller_memory.remembered_language({"language": "bn-IN"}) is None, "non-reply language ignored")
check(caller_memory.remembered_language({}) is None, "missing language ignored")

print()
if failures:
    print(f"{len(failures)} FAILURES")
    sys.exit(1)
print("ALL PASSED")
sys.exit(0)
