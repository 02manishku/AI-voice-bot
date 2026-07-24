"""In-memory rate limiting.

Every turn spends real money (~Rs 0.6, mostly TTS characters), so the point of
this is credit protection, not abuse protection: a client stuck in a retry loop,
or a forgotten open tab, can drain the account overnight. Per-caller limits stop
one bad session; the daily cap bounds total exposure.

In-memory and per-process, which is right for a single-process demo and wrong
for anything multi-worker. Deliberate: §11 rules out a database.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass

MINUTE = 60.0
DAY = 24 * 60 * 60.0


@dataclass
class Decision:
    allowed: bool
    message: str = ""
    retry_after: int = 0


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int):
        self.per_minute = per_minute
        self.per_day = per_day
        self._callers: dict[str, deque[float]] = defaultdict(deque)
        self._day: deque[float] = deque()

    @staticmethod
    def _trim(stamps: deque[float], window: float, now: float) -> None:
        while stamps and now - stamps[0] > window:
            stamps.popleft()

    def check(self, caller: str) -> Decision:
        """Test and record in one step — callers must not be able to forget."""
        now = time.monotonic()

        self._trim(self._day, DAY, now)
        if len(self._day) >= self.per_day:
            return Decision(
                False,
                f"Daily limit reached ({self.per_day} calls). This guard exists so a "
                f"runaway client can't drain the Sarvam credits. Raise "
                f"MAX_TURNS_PER_DAY in .env to lift it.",
                retry_after=int(DAY - (now - self._day[0])),
            )

        mine = self._callers[caller]
        self._trim(mine, MINUTE, now)
        if len(mine) >= self.per_minute:
            wait = int(MINUTE - (now - mine[0])) + 1
            return Decision(
                False,
                f"Slow down a moment — that's {self.per_minute} questions in under a "
                f"minute. Try again in {wait}s.",
                retry_after=wait,
            )

        mine.append(now)
        self._day.append(now)

        # Don't let idle callers accumulate forever.
        if len(self._callers) > 512:
            for key in [k for k, v in self._callers.items() if not v]:
                del self._callers[key]

        return Decision(True)

    def spent_today(self) -> int:
        self._trim(self._day, DAY, time.monotonic())
        return len(self._day)
