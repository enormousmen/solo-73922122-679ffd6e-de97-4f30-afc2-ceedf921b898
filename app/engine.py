"""Single-process micro-batch inference engine with fault isolation.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch (or recovery probe) is
  ever actually inferring at any moment.
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
queued/processing -> failed    (the batch raised; fault isolation, see below)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.

Fault isolation
---------------
When a batch raises:
* the engine enters `isolated` BEFORE the HTTP layer can see the failure, so
  every subsequent POST /predict gets 503 while the model is suspect;
* every order of the failed batch that is still `processing` becomes `failed`;
* every order still `queued` at that moment (not only orders that happened to
  be copied into the failed batch) becomes `failed` as well, and capacity is
  released for both groups;
* orders already `cancelled` or `expired` keep their state and their result
  slot; they were never going to yield a result.
Orders in any state remain queryable and terminal orders stay cancellable
(no-op, 200 with their unchanged state).

Recovery probe
--------------
`probe()` runs ONE synthetic 64x64 RGB input through the very same model and
executor thread that serve batches. A probe succeeds only if the model returns
exactly 10 finite logits. Success closes the isolation gate and accepts new
work again; failure keeps the gate closed and the probe may be retried.
Only one probe can run at a time (a concurrent call gets `already_running`).
The probe never blocks the event loop, so queries/cancels stay responsive.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

from .preprocessing import recovery_probe_input

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"

# A frame is failed by fault isolation only if it was still waiting for a
# result. Cancelled/expired orders keep their state untouched.
_FAILABLE_STATES = (QUEUED, PROCESSING)

NUM_CLASSES = 10


@dataclass(eq=False)
class ProbeResult:
    """Outcome of one recovery probe (no tensors kept: slots stay small)."""

    id: str
    started_at: float
    finished_at: float = 0.0
    ok: bool = False
    # "passed" | "bad_shape" | "non_finite" | "exception" | "cancelled"
    reason: str = "cancelled"
    detail: str = ""
    logits_shape: Optional[list[int]] = None

    def as_dict(self) -> dict:
        return {
            "probe_id": self.id,
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
            "logits_shape": self.logits_shape,
            "started_at": round(self.started_at, 3),
            "finished_at": round(self.finished_at, 3),
            "duration_ms": round((self.finished_at - self.started_at) * 1000),
        }


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    # Set when the batch this order rode in raised. Lets the HTTP layer tell
    # a model fault apart from a deadline expiry for any state.
    batch_failed: bool = False


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
        inject_failure_count: int = 0,
        probe_delay_ms: int = 0,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0
        self.probe_delay_s = probe_delay_ms / 1000.0
        # Demo fault injection: the next N inference calls raise once.
        self._failures_remaining = max(0, inject_failure_count)

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders inside the batch currently being computed
        self._not_empty = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None
        # Fault isolation + recovery probe.
        self._isolated = False
        self._probe_active = False
        self._probe_task: Optional[asyncio.Task] = None
        self.last_probe: Optional[ProbeResult] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0
        self.failed_batches = 0
        self.probes_run = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        # Pay the first-call TorchScript/kernel-loading cost before traffic so
        # the event loop is not stalled by it during the first real batch.
        # The warm-up never consumes the demo fault injection.
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._warmup)
        except Exception:  # noqa: BLE001 - a broken model surfaces on real work
            log.exception("model warm-up failed")
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._probe_task is not None:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - shutdown must not raise
                log.exception("recovery probe failed during shutdown")
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def is_isolated(self) -> bool:
        return self._isolated

    def probe_running(self) -> bool:
        return self._probe_active

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "accepting_orders": not self._isolated,
            "isolated": self._isolated,
            "probe_running": self.probe_running(),
            "last_probe": self.last_probe.as_dict() if self.last_probe else None,
            "failed_batches": self.failed_batches,
            "probes_run": self.probes_run,
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> tuple[Optional[Order], str]:
        """Accept an image, or explain why not.

        Returns ``(order, "")`` on success, ``(None, "isolated")`` while the
        fault gate is closed and ``(None, "capacity")`` when all 16 slots are
        taken. Isolation wins over capacity: while the model is suspect
        nothing new may enter, whatever the free slots.
        """
        if self._isolated:
            return None, "isolated"
        if self.capacity_used() >= self.max_capacity:
            return None, "capacity"
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._not_empty.set()
        return order, ""

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        # completed / expired / cancelled / failed: terminal, returned as-is.
        return o

    # ------------------------------------------------------------------ #
    # fault isolation
    # ------------------------------------------------------------------ #
    def _trip_breaker(self, batch_id: str, batch: list[Order]) -> None:
        """Enter isolation and fail the batch + everything still queued.

        Runs on the event loop right after the failing batch returns, before
        the worker can start another batch or the HTTP layer can observe the
        failure, so no frame can slip into a second bad batch.

        Orders already cancelled/expired keep their state. Orders whose
        deadline passed by the time the failure is handled are settled as
        expired (same rule as a late successful result). Everything else that
        was waiting for a result becomes failed.
        """
        self._isolated = True
        self.failed_batches += 1
        # `batch` holds the frames whose compute raised (they were set to
        # PROCESSING) plus, on a formation race, any the queue concurrently
        # handed out. `_pending` holds every frame that did not reach compute.
        in_batch = {id(o) for o in batch}
        doomed = {id(o): o for o in batch}
        for o in list(self._pending):
            doomed[id(o)] = o
        for o in doomed.values():
            # Settle deadline expiry first: expired frames keep their state.
            self._refresh_state(o)
            if o.state in _FAILABLE_STATES:
                # Only frames that actually reached the model get a batch id;
                # queued frames are failed without ever having been batched.
                if id(o) in in_batch:
                    o.batch_id = batch_id
                o.batch_failed = True
                o.state = FAILED
                o.result = None
        # Every failed frame is terminal; release all of its capacity at once,
        # including the batch that was in-flight. Cancelled/expired frames were
        # never part of either collection.
        self._pending.clear()
        self._inflight = 0
        self._not_empty.clear()
        log.warning(
            "model isolated: batch %s failed; %d order(s) settled",
            batch_id,
            len(doomed),
        )

    def _check_logits(self, logits: torch.Tensor) -> Optional[str]:
        """Return None if the model output is acceptable, else a reason."""
        shape = list(logits.shape)
        if len(shape) != 2 or shape[0] != 1 or shape[1] != NUM_CLASSES:
            return f"expected shape [1, {NUM_CLASSES}], got {shape}"
        if not bool(torch.isfinite(logits).all()):
            return "logits contain NaN or Inf"
        return None

    async def probe(self) -> ProbeResult:
        """Run one recovery probe on the batch inference channel.

        Returns when the probe has completed (success or failure). Only one
        probe may run at a time; a concurrent caller gets a finished
        result with reason ``already_running`` and does not touch the model.
        The flag is set synchronously before the first await so two requests
        in the same event-loop tick cannot both start a probe.

        The probe task is shielded: if the HTTP client disconnects or times
        out, the probe still finishes and its verdict is still applied, so
        an abandoned request can never leave the gate half-open or let a
        second probe race the abandoned one.
        """
        if self._probe_active:
            return ProbeResult(
                id=uuid.uuid4().hex[:8],
                started_at=time.monotonic(),
                finished_at=time.monotonic(),
                ok=False,
                reason="already_running",
                detail="a recovery probe is already in progress",
            )
        self._probe_active = True
        self._probe_task = asyncio.ensure_future(self._run_probe_guarded())
        return await asyncio.shield(self._probe_task)

    async def _run_probe_guarded(self) -> ProbeResult:
        try:
            return await self._run_probe()
        except asyncio.CancelledError:
            # Only shutdown cancels the probe (requests are shielded).
            self.last_probe = ProbeResult(
                id=uuid.uuid4().hex[:8],
                started_at=time.monotonic(),
                finished_at=time.monotonic(),
                ok=False,
                reason="cancelled",
                detail="probe cancelled during shutdown",
            )
            raise
        finally:
            self._probe_active = False
            self._probe_task = None

    async def _run_probe(self) -> ProbeResult:
        probe_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        # Deterministic, valid 64x64 RGB input; same normalisation as frames.
        tensor = recovery_probe_input()
        try:
            logits = await loop.run_in_executor(
                self._executor, self._probe_forward, tensor
            )
        except Exception as exc:  # noqa: BLE001 - probe failure must not escape
            log.warning("recovery probe %s raised: %s", probe_id, exc)
            res = ProbeResult(
                id=probe_id,
                started_at=started,
                finished_at=time.monotonic(),
                ok=False,
                reason="exception",
                detail=f"{type(exc).__name__}: {exc}",
            )
        else:
            shape = list(logits.shape)
            reason = self._check_logits(logits)
            if reason is None:
                res = ProbeResult(
                    id=probe_id,
                    started_at=started,
                    finished_at=time.monotonic(),
                    ok=True,
                    reason="passed",
                    detail=f"all {NUM_CLASSES} logits finite",
                    logits_shape=shape,
                )
            else:
                res = ProbeResult(
                    id=probe_id,
                    started_at=started,
                    finished_at=time.monotonic(),
                    ok=False,
                    reason="bad_shape" if "shape" in reason else "non_finite",
                    detail=reason,
                    logits_shape=shape,
                )

        self.probes_run += 1
        self.last_probe = res
        if res.ok:
            self._isolated = False
            log.warning("recovery probe %s passed; accepting orders again", probe_id)
        return res

    # ------------------------------------------------------------------ #
    # batching worker
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._not_empty.wait()
            if self._isolated:
                # Nothing may enter the model while isolated. New submissions
                # are rejected at the HTTP layer, so this is only defensive.
                self._not_empty.clear()
                continue

            # Batch-formation window: the idle model waits at most
            # `batch_wait_s` for more work, up to `batch_size` images.
            batch: list[Order] = []
            window_end = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                self._purge_expired_pending()
                while self._pending and len(batch) < self.batch_size:
                    batch.append(self._pending.popleft())
                if len(batch) >= self.batch_size:
                    break
                remaining = window_end - loop.time()
                if remaining <= 0:
                    break
                self._not_empty.clear()
                # Re-check after clearing: a submit that landed between the
                # pop and the clear must not be lost to a missed wakeup.
                if self._pending:
                    continue
                try:
                    await asyncio.wait_for(self._not_empty.wait(), remaining)
                except asyncio.TimeoutError:
                    break

            # Final gate: cancelled/expired orders never enter the model.
            now = time.monotonic()
            batch = [o for o in batch if o.state == QUEUED and o.deadline_at > now]
            if not batch:
                continue
            if self._isolated:
                # Defensive: a batch failure cannot tear down the queue while
                # the worker is idle, but if it ever did, never re-run frames
                # for a suspect model — settle them as failed.
                for o in batch:
                    self._refresh_state(o)
                    if o.state in _FAILABLE_STATES:
                        o.batch_failed = True
                        o.state = FAILED
                continue

            batch_id = uuid.uuid4().hex[:8]
            for o in batch:
                o.state = PROCESSING
                o.batch_id = batch_id
            self._inflight = len(batch)
            self.batches_started += 1
            self.last_batch_size = len(batch)
            tensors = torch.stack([o.tensor for o in batch])

            try:
                scores = await loop.run_in_executor(self._executor, self._infer, tensors)
            except Exception:  # noqa: BLE001 - model failure must not kill the worker
                log.exception("inference failed for batch %s", batch_id)
                self._trip_breaker(batch_id, batch)
                continue

            now = time.monotonic()
            for o, sc in zip(batch, scores):
                if o.state != PROCESSING:
                    # Cancelled or expired mid-flight: result is discarded and
                    # the late result must NOT revive the order.
                    continue
                if o.deadline_at <= now:
                    o.state = EXPIRED
                    continue
                o.state = COMPLETED
                predicted = max(range(len(sc)), key=lambda i: sc[i])
                o.result = {"predicted_class": predicted, "scores": sc}
            self._inflight = 0

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _inject_failure(self) -> None:
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise RuntimeError("injected inference failure (demo fault injection)")

    def _warmup(self) -> None:
        """Load kernels/weights once; not a demo failure opportunity."""
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, 64, 64))

    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        self._inject_failure()
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]

    def _probe_forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.probe_delay_s > 0:
            time.sleep(self.probe_delay_s)  # demo: simulate a slow probe
        self._inject_failure()
        with torch.inference_mode():
            return self.model(tensor)
