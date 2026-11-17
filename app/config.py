"""Runtime configuration, overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "model.pt")


def _env_failure_count() -> int:
    """Read the demo fault-injection knob.

    ``INJECT_FAILURE_ONCE`` accepts booleans (``true`` -> 1) or a positive
    integer.  With N > 0 the next N inference calls (a batch or a recovery
    probe) raise once; N=1 is the normal one-shot demo.  N=2 is useful to show
    that the first recovery probe can also fail and that the probe is retryable.
    ``INJECT_FAILURE_COUNT`` is accepted as an explicit numeric alias.
    """
    raw = os.getenv("INJECT_FAILURE_ONCE")
    if raw is None or raw.strip() == "":
        raw = os.getenv("INJECT_FAILURE_COUNT", "0")
    value = raw.strip().lower()
    if value in ("true", "yes", "on"):
        return 1
    if value in ("false", "no", "off", ""):
        return 0
    return max(0, int(value))


@dataclass
class Settings:
    # Hard limits from the spec (can be lowered in tests).
    max_capacity: int = 16        # max images queued OR in-flight
    batch_size: int = 4           # max images per inference batch
    batch_wait_ms: int = 50       # idle model waits at most this long to fill a batch
    # Demo knob: artificial delay inside the inference worker to simulate a slow model.
    inference_delay_ms: int = 0
    # Demo knob: fail the next N inference calls (batches and/or probes) once.
    inject_failure_count: int = 0
    # Demo knob: artificial delay inside the recovery probe (simulates a slow probe).
    probe_delay_ms: int = 0
    model_path: str = _DEFAULT_MODEL_PATH

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_capacity=int(os.getenv("MAX_CAPACITY", "16")),
            batch_size=int(os.getenv("BATCH_SIZE", "4")),
            batch_wait_ms=int(os.getenv("BATCH_WAIT_MS", "50")),
            inference_delay_ms=int(os.getenv("INFERENCE_DELAY_MS", "0")),
            inject_failure_count=_env_failure_count(),
            probe_delay_ms=int(os.getenv("PROBE_DELAY_MS", "0")),
            model_path=os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH),
        )
