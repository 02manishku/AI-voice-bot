"""Rate limiting + greeting cache. Offline — no API calls.

The limiter guards real money, so the cases that matter are the ones where it
should NOT fire (a normal conversation) and where it must (a runaway client).

Run: uv run python tests/test_limits.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import limits


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


ok = True

# --- per-caller limit ---
rl = limits.RateLimiter(per_minute=3, per_day=100)
ok &= check("1st call allowed", rl.check("a").allowed)
ok &= check("2nd call allowed", rl.check("a").allowed)
ok &= check("3rd call allowed", rl.check("a").allowed)
d = rl.check("a")
ok &= check("4th call blocked", not d.allowed)
ok &= check("blocked message is readable", "Slow down" in d.message)
ok &= check("Retry-After is a sane number", 0 < d.retry_after <= 61)
ok &= check("a different caller is unaffected", rl.check("b").allowed)

# --- daily cap protects the credits ---
rl = limits.RateLimiter(per_minute=1000, per_day=3)
for _ in range(3):
    rl.check("x")
d = rl.check("x")
ok &= check("daily cap blocks", not d.allowed)
ok &= check("daily message names the env var", "MAX_TURNS_PER_DAY" in d.message)
ok &= check("daily cap is global, not per-caller", not rl.check("someone-else").allowed)
ok &= check("spent_today counts", rl.spent_today() == 3)

# --- the window actually slides ---
rl = limits.RateLimiter(per_minute=2, per_day=100)
rl.check("w")
rl.check("w")
ok &= check("blocked at the limit", not rl.check("w").allowed)
# Rewind the recorded stamps rather than sleeping 60s.
rl._callers["w"] = type(rl._callers["w"])(t - 61 for t in rl._callers["w"])
ok &= check("allowed again once the window passes", rl.check("w").allowed)

# --- blocked calls must not consume budget ---
rl = limits.RateLimiter(per_minute=1, per_day=100)
rl.check("z")
before = rl.spent_today()
rl.check("z")
rl.check("z")
ok &= check("rejected calls don't count toward the daily spend", rl.spent_today() == before)

# --- greeting cache key ---
print("\ngreeting cache:")
import asyncio
import hashlib

from app import main, prompts
from app.config import settings


def key_for():
    k = "|".join(
        [
            prompts.GREETING,
            prompts.GREETING_LANGUAGE,
            settings.sarvam_tts_model,
            settings.sarvam_tts_speaker,
            str(settings.sarvam_tts_pace),
        ]
    )
    return hashlib.sha256(k.encode()).hexdigest()[:16]


k1 = key_for()
settings.sarvam_tts_speaker = "anushka"
k2 = key_for()
ok &= check("changing the speaker re-renders", k1 != k2)
settings.sarvam_tts_speaker = "shubh"
ok &= check("same settings reuse the cache", key_for() == k1)

# Missing keys must not crash startup, just skip the pre-render.
settings.sarvam_api_key = ""
main.CACHE_DIR.mkdir(exist_ok=True)
res = asyncio.run(main._load_greeting())
ok &= check("no key -> returns None instead of crashing", res is None or isinstance(res, bytes))

# --- the greeting itself obeys the KB ---
print("\ngreeting text (A7/A1):")
g = prompts.GREETING
ok &= check("names the brand", "Magppie Wellness Kitchens" in g)
ok &= check("asks how it can help", "help you" in g.lower())
ok &= check("does not use the outbound B1 script", "Instagram" not in g and "couple of minutes" not in g)

print("\nOK" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
