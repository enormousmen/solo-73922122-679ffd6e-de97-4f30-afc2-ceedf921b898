"""End-to-end tests for the micro-batch classifier API.

Each test gets a fresh app/engine via the `client` factory. Inference delay
is injected through Settings to simulate a slow model without sleeping the
event loop.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
import torch
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"
SAMPLE_NAMES = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png", "sample_noise.png"]


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(MODEL))
    return Settings(**kw)


class Client:
    """httpx client wrapper with helpers."""

    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app

    async def submit(self, name: str, deadline_ms: int = 60_000):
        data = (SAMPLES / name).read_bytes()
        return await self.http.post(
            "/predict",
            files={"file": (name, data, "image/png")},
            data={"deadline_ms": str(deadline_ms)},
        )

    async def get(self, order_id: str):
        return (await self.http.get(f"/orders/{order_id}")).json()

    async def cancel(self, order_id: str):
        return await self.http.post(f"/orders/{order_id}/cancel")

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# basic flow
# --------------------------------------------------------------------- #
async def test_single_prediction_completes(make_client):
    c = await make_client()
    r = await c.submit("sample_gradient.png", deadline_ms=5000)
    assert r.status_code == 201, r.text
    oid = r.json()["order_id"]
    body = await c.wait_state(oid, {"completed"})
    scores = body["result"]["scores"]
    assert len(scores) == 10
    assert abs(sum(scores) - 1.0) < 1e-4
    assert 0 <= body["result"]["predicted_class"] < 10


async def test_validation_and_404(make_client):
    c = await make_client()
    # non-positive deadline rejected
    r = await c.submit("sample_gradient.png", deadline_ms=0)
    assert r.status_code == 400
    # garbage bytes rejected
    r = await c.http.post(
        "/predict",
        files={"file": ("x.bin", b"not an image", "application/octet-stream")},
        data={"deadline_ms": "1000"},
    )
    assert r.status_code == 400
    # unknown order
    assert (await c.http.get("/orders/nope")).status_code == 404
    assert (await c.http.post("/orders/nope/cancel")).status_code == 404


# --------------------------------------------------------------------- #
# batching correctness: batched scores match per-image inference <= 1e-5
# --------------------------------------------------------------------- #
async def test_batch_scores_match_single_inference(make_client):
    c = await make_client()

    # reference: each image inferred alone
    single_scores = {}
    for name in SAMPLE_NAMES:
        r = await c.submit(name)
        body = await c.wait_state(r.json()["order_id"], {"completed"})
        single_scores[name] = body["result"]["scores"]

    # batched: 4 concurrent submissions must ride one batch of 4
    responses = await asyncio.gather(*[c.submit(n) for n in SAMPLE_NAMES])
    ids = [r.json()["order_id"] for r in responses]
    bodies = await asyncio.gather(*[c.wait_state(i, {"completed"}) for i in ids])
    batch_ids = {b["batch_id"] for b in bodies}
    assert len(batch_ids) == 1, f"expected one shared batch, got {batch_ids}"
    assert c.app.state.engine.last_batch_size == 4

    for name, body in zip(SAMPLE_NAMES, bodies):
        for a, b in zip(body["result"]["scores"], single_scores[name]):
            assert abs(a - b) <= 1e-5, f"{name}: {a} vs {b}"


async def test_idle_model_waits_to_fill_batch(make_client):
    c = await make_client(batch_wait_ms=500)
    r1 = await c.submit("sample_gradient.png")
    await asyncio.sleep(0.05)  # inside the batch window
    r2 = await c.submit("sample_circles.png")
    b1 = await c.wait_state(r1.json()["order_id"], {"completed"})
    b2 = await c.wait_state(r2.json()["order_id"], {"completed"})
    assert b1["batch_id"] == b2["batch_id"]
    assert c.app.state.engine.last_batch_size == 2


# --------------------------------------------------------------------- #
# capacity: 16 queued+in-flight, then 429
# --------------------------------------------------------------------- #
async def test_capacity_full_rejects_429(make_client):
    c = await make_client(inference_delay_ms=1500)
    accepted = []
    for i in range(16):
        r = await c.submit(SAMPLE_NAMES[i % 4])
        assert r.status_code == 201, (i, r.text)
        accepted.append(r.json()["order_id"])
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 429, r.text
    stats = (await c.http.get("/stats")).json()
    assert stats["capacity_used"] == 16


# --------------------------------------------------------------------- #
# cancellation
# --------------------------------------------------------------------- #
async def test_cancel_queued_order_never_enters_model(make_client):
    c = await make_client(inference_delay_ms=1200)
    ids = []
    for i in range(5):  # 4 fill the running batch, 5th stays queued
        r = await c.submit(SAMPLE_NAMES[i % 4])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # let the first batch start
    victim = ids[4]
    body = await c.get(victim)
    assert body["status"] == "queued"
    r = await c.cancel(victim)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    # after everything settles it must still be cancelled, with no result/batch
    await asyncio.sleep(1.6)
    body = await c.get(victim)
    assert body["status"] == "cancelled"
    assert body["result"] is None
    assert body["batch_id"] is None


async def test_cancel_processing_marks_now_but_holds_capacity(make_client):
    c = await make_client(inference_delay_ms=1500)
    ids = []
    for i in range(4):  # exactly one batch
        r = await c.submit(SAMPLE_NAMES[i])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # batch is now computing
    victim, others = ids[0], ids[1:]
    assert (await c.get(victim))["status"] == "processing"

    r = await c.cancel(victim)
    assert r.json()["status"] == "cancelled"  # marked immediately
    assert (await c.get(victim))["status"] == "cancelled"

    # slot is still occupied while the batch computes: only 12 free slots
    extra = []
    for _ in range(12):
        r = await c.submit("sample_gradient.png")
        assert r.status_code == 201
        extra.append(r.json()["order_id"])
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 429, "cancelled in-flight order must still occupy capacity"

    # no new batch may start while the cancelled one's batch is computing
    assert c.app.state.engine.batches_started == 1

    # after compute ends: victim stays cancelled without result, others complete
    await asyncio.sleep(2.2)
    body = await c.get(victim)
    assert body["status"] == "cancelled" and body["result"] is None
    for oid in others:
        assert (await c.get(oid))["status"] == "completed"


async def test_cancel_completed_order_conflicts(make_client):
    c = await make_client()
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await c.wait_state(oid, {"completed"})
    r = await c.cancel(oid)
    assert r.status_code == 409


# --------------------------------------------------------------------- #
# expiry
# --------------------------------------------------------------------- #
async def test_queued_order_expires_and_never_enters_model(make_client):
    c = await make_client(inference_delay_ms=1200)
    ids = []
    for i in range(4):  # occupy the model with one batch
        r = await c.submit(SAMPLE_NAMES[i])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)
    r = await c.submit("sample_gradient.png", deadline_ms=300)  # expires while queued
    victim = r.json()["order_id"]
    await asyncio.sleep(0.5)
    body = await c.get(victim)
    assert body["status"] == "expired"
    await asyncio.sleep(1.5)  # let the running batch finish and next batch form
    body = await c.get(victim)
    assert body["status"] == "expired"
    assert body["result"] is None
    assert body["batch_id"] is None  # never entered the model


async def test_processing_order_expires_midflight_no_result(make_client):
    c = await make_client(inference_delay_ms=1000)
    r = await c.submit("sample_gradient.png", deadline_ms=400)
    oid = r.json()["order_id"]
    await asyncio.sleep(1.6)  # batch computed (late), deadline long past
    body = await c.get(oid)
    assert body["status"] == "expired"
    assert body["result"] is None


async def test_late_result_does_not_revive_cancelled_order(make_client):
    c = await make_client(inference_delay_ms=800)
    r = await c.submit("sample_gradient.png", deadline_ms=60_000)
    oid = r.json()["order_id"]
    await asyncio.sleep(0.2)  # in-flight now
    await c.cancel(oid)
    await asyncio.sleep(1.2)  # batch compute finished, result arrived late
    body = await c.get(oid)
    assert body["status"] == "cancelled"
    assert body["result"] is None


# --------------------------------------------------------------------- #
# slow inference must not block submit / query / cancel
# --------------------------------------------------------------------- #
async def test_slow_inference_does_not_block_api(make_client):
    c = await make_client(inference_delay_ms=2000)
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await asyncio.sleep(0.2)  # batch is computing for ~2s

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    (resp, dt1) = await timed(c.http.get(f"/orders/{oid}"))
    (resp2, dt2) = await timed(c.submit("sample_circles.png"))
    (resp3, dt3) = await timed(c.cancel(resp2.json()["order_id"]))
    (resp4, dt4) = await timed(c.http.get("/stats"))
    for dt in (dt1, dt2, dt3, dt4):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during slow inference"
    assert resp.status_code == 200
    assert resp2.status_code == 201
    assert resp3.status_code == 200
    assert resp4.status_code == 200


# --------------------------------------------------------------------- #
# fault isolation: a failed batch trips the gate, new work gets 503
# --------------------------------------------------------------------- #
async def test_failed_batch_isolates_and_fails_batch_and_queue(make_client):
    c = await make_client(inference_delay_ms=500, inject_failure_count=1)
    batch_ids = []
    for i in range(4):  # exactly one full batch; its compute will raise
        r = await c.submit(SAMPLE_NAMES[i])
        assert r.status_code == 201, r.text
        batch_ids.append(r.json()["order_id"])
    await asyncio.sleep(0.15)  # first batch is now computing
    for oid in batch_ids:
        assert (await c.get(oid))["status"] == "processing"

    # orders that will be waiting when the failure happens
    queued = (await c.submit(SAMPLE_NAMES[0])).json()["order_id"]
    cancelled = (await c.submit(SAMPLE_NAMES[1])).json()["order_id"]
    assert (await c.cancel(cancelled)).json()["status"] == "cancelled"
    expired = (await c.submit(SAMPLE_NAMES[2], deadline_ms=100)).json()["order_id"]

    await asyncio.sleep(0.55)  # batch raises; breaker must have tripped

    # in-flight frames of the failing batch -> failed, no result
    for oid in batch_ids:
        body = await c.get(oid)
        assert body["status"] == "failed", body
        assert body["result"] is None
        assert body["failure"] == "inference_error"
        assert body["batch_id"] is not None

    # frames still queued when the model failed -> failed, never batched
    body = await c.get(queued)
    assert body["status"] == "failed", body
    assert body["result"] is None
    assert body["failure"] == "inference_error"
    assert body["batch_id"] is None

    # cancelled/expired keep their state and are not marked as model failures
    body = await c.get(cancelled)
    assert body["status"] == "cancelled" and body["failure"] is None
    body = await c.get(expired)
    assert body["status"] == "expired" and body["failure"] is None

    # capacity released, gate closed, new submissions rejected with 503
    stats = (await c.http.get("/stats")).json()
    assert stats["capacity_used"] == 0
    assert stats["isolated"] is True
    assert stats["accepting_orders"] is False
    assert stats["failed_batches"] == 1
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 503, r.text
    assert "recovery/probe" in r.json()["detail"]

    # the previously failed/cancelled/expired orders stay queryable
    assert (await c.http.get(f"/orders/{queued}")).status_code == 200
    assert (await c.http.get(f"/orders/{expired}")).status_code == 200


# --------------------------------------------------------------------- #
# recovery probe: passes -> accept again; fails -> stay isolated, retry
# --------------------------------------------------------------------- #
async def test_recovery_probe_passes_and_reopens(make_client):
    c = await make_client(inject_failure_count=1)
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await c.wait_state(oid, {"failed"})
    assert (await c.submit("sample_gradient.png")).status_code == 503

    r = await c.http.post("/recovery/probe")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recovered"] is True
    assert body["isolated"] is False
    assert body["probe"]["logits_shape"] == [1, 10]
    assert body["probe"]["reason"] == "passed"

    status = (await c.http.get("/recovery")).json()
    assert status["isolated"] is False and status["accepting_orders"] is True
    assert status["probe_running"] is False
    assert (await c.http.get("/stats")).json()["probes_run"] == 1

    # service accepts and completes work again
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 201, r.text
    body = await c.wait_state(r.json()["order_id"], {"completed"})
    assert len(body["result"]["scores"]) == 10


async def test_failed_probe_keeps_isolation_and_is_retryable(make_client):
    # two injected failures: first the batch, then the first probe
    c = await make_client(inject_failure_count=2)
    r = await c.submit("sample_gradient.png")
    await c.wait_state(r.json()["order_id"], {"failed"})
    assert (await c.submit("sample_gradient.png")).status_code == 503

    r = await c.http.post("/recovery/probe")
    assert r.status_code == 503, r.text
    assert r.json()["recovered"] is False
    assert r.json()["isolated"] is True
    assert r.json()["probe"]["reason"] == "exception"
    assert (await c.submit("sample_gradient.png")).status_code == 503

    # retry succeeds and only then reopens the gate
    r = await c.http.post("/recovery/probe")
    assert r.status_code == 200, r.text
    assert (await c.submit("sample_gradient.png")).status_code == 201


async def test_concurrent_probes_never_run_in_parallel(make_client):
    c = await make_client(inject_failure_count=1, probe_delay_ms=600)
    r = await c.submit("sample_gradient.png")
    await c.wait_state(r.json()["order_id"], {"failed"})

    responses = await asyncio.gather(
        c.http.post("/recovery/probe"), c.http.post("/recovery/probe")
    )
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 409], [r.text for r in responses]
    conflict = next(r for r in responses if r.status_code == 409)
    assert conflict.json()["probe"]["reason"] == "already_running"
    assert c.app.state.engine.probes_run == 1
    assert (await c.http.get("/stats")).json()["probe_running"] is False


async def test_slow_probe_does_not_block_queries_or_cancel(make_client):
    c = await make_client(inject_failure_count=1, probe_delay_ms=1200)
    r = await c.submit("sample_gradient.png")
    failed_id = r.json()["order_id"]
    await c.wait_state(failed_id, {"failed"})

    probe_task = asyncio.create_task(c.http.post("/recovery/probe"))
    await asyncio.sleep(0.1)  # probe now computing in the inference thread
    assert (await c.http.get("/stats")).json()["probe_running"] is True

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    (status, dt1) = await timed(c.http.get("/recovery"))
    (order, dt2) = await timed(c.http.get(f"/orders/{failed_id}"))
    (cancel, dt3) = await timed(c.http.post(f"/orders/{failed_id}/cancel"))
    (stats, dt4) = await timed(c.http.get("/stats"))
    (new_work, dt5) = await timed(c.submit("sample_gradient.png"))
    for dt in (dt1, dt2, dt3, dt4, dt5):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during slow probe"
    assert status.status_code == 200
    assert order.json()["status"] == "failed"  # old order still queryable
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "failed"  # terminal state preserved
    assert stats.json()["probe_running"] is True
    # still isolated until the probe actually succeeds
    assert new_work.status_code == 503

    probe_resp = await probe_task
    assert probe_resp.status_code == 200


async def test_probe_rejects_non_finite_or_wrong_shape(make_client):
    c = await make_client(inject_failure_count=1)
    r = await c.submit("sample_gradient.png")
    await c.wait_state(r.json()["order_id"], {"failed"})
    engine = c.app.state.engine
    original = engine.model

    # 10 logits but non-finite -> must not reopen the gate
    engine.model = lambda t: torch.full((1, 10), float("nan"))
    try:
        r = await c.http.post("/recovery/probe")
        assert r.status_code == 503, r.text
        assert r.json()["probe"]["reason"] == "non_finite"
        assert r.json()["probe"]["logits_shape"] == [1, 10]
        assert (await c.submit("sample_gradient.png")).status_code == 503

        # wrong shape -> must not reopen the gate
        engine.model = lambda t: torch.zeros((4, 10))
        r = await c.http.post("/recovery/probe")
        assert r.status_code == 503, r.text
        assert r.json()["probe"]["reason"] == "bad_shape"
        assert (await c.submit("sample_gradient.png")).status_code == 503
    finally:
        engine.model = original

    # the real model passes and reopens the gate
    r = await c.http.post("/recovery/probe")
    assert r.status_code == 200, r.text


async def test_recovery_probe_available_when_healthy(make_client):
    """Probe is also callable before any failure (e.g. canary); it must not disrupt."""
    c = await make_client()
    r = await c.http.post("/recovery/probe")
    assert r.status_code == 200, r.text
    assert r.json()["probe"]["logits_shape"] == [1, 10]
    assert (await c.http.get("/recovery")).json()["isolated"] is False
    # normal work is unaffected
    r = await c.submit("sample_gradient.png")
    assert (await c.wait_state(r.json()["order_id"], {"completed"}))["status"] == "completed"


async def test_injected_failure_is_one_shot_by_default(make_client):
    """INJECT_FAILURE_ONCE=true fails exactly one call; the service then recovers."""
    c = await make_client(inject_failure_count=1)
    first = await c.submit("sample_gradient.png")
    await c.wait_state(first.json()["order_id"], {"failed"})
    assert (await c.http.post("/recovery/probe")).status_code == 200
    # second batch is not failed by the demo knob
    second = await c.submit("sample_circles.png")
    assert (await c.wait_state(second.json()["order_id"], {"completed"}))["status"] == "completed"


async def test_probe_input_is_one_64x64_rgb_image(make_client):
    c = await make_client(inject_failure_count=1)
    r = await c.submit("sample_gradient.png")
    await c.wait_state(r.json()["order_id"], {"failed"})
    engine = c.app.state.engine
    original = engine.model
    seen = {}

    def recording_model(t):
        assert tuple(t.shape) == (1, 3, 64, 64), tuple(t.shape)
        seen["value_range"] = (float(t.min()), float(t.max()))
        return torch.zeros((1, 10))

    engine.model = recording_model
    try:
        r = await c.http.post("/recovery/probe")
        assert r.status_code == 200, r.text
        assert r.json()["probe"]["logits_shape"] == [1, 10]
    finally:
        engine.model = original
