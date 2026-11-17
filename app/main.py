"""FastAPI HTTP layer for the micro-batch image classifier.

Endpoints
---------
POST /predict                    submit an image + deadline_ms (from acceptance)
GET  /orders/{order_id}          query status / result by order id
POST /orders/{order_id}/cancel   cancel a queued or in-flight order
POST /recovery/probe             run a recovery probe against the model
GET  /recovery                   isolation / probe status
GET  /stats                      capacity and state counters (demo/observability)
GET  /health                     liveness
Interactive docs are served at /docs (Swagger UI).

Fault isolation contract
------------------------
* While isolated, POST /predict returns 503 and no frame reaches the model.
* A failed batch and every order queued at that moment become `failed` and
  release their capacity. Already cancelled/expired orders keep their state.
* POST /recovery/probe returns 200 only when the model produced exactly 10
  finite logits for one 64x64 RGB input; only then are new orders accepted.
  A failed probe returns 503 and may be retried. Concurrent probes are
  refused with 409. Queries and cancels keep working during a slow probe.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings
from .engine import COMPLETED, BatchEngine, Order
from .preprocessing import preprocess

ISOLATED_DETAIL = (
    "model isolated after an inference failure; "
    "retry POST /recovery/probe until it succeeds"
)


def order_view(o: Order) -> dict:
    remaining_ms = max(0, round((o.deadline_at - time.monotonic()) * 1000))
    return {
        "order_id": o.id,
        "status": o.state,
        "deadline_ms": o.deadline_ms,
        "remaining_ms": remaining_ms,
        "batch_id": o.batch_id,
        "result": o.result if o.state == COMPLETED else None,
        # True when the inference error that tripped fault isolation hit this
        # order's batch (or the queue it was waiting in).
        "failure": "inference_error" if o.batch_failed else None,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = BatchEngine(
        model_path=settings.model_path,
        max_capacity=settings.max_capacity,
        batch_size=settings.batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        inference_delay_ms=settings.inference_delay_ms,
        inject_failure_count=settings.inject_failure_count,
        probe_delay_ms=settings.probe_delay_ms,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        await engine.stop()

    app = FastAPI(
        title="Micro-batch Image Classifier",
        version="1.1.0",
        description=(
            "Time-limited image classification for multiple cameras. "
            "Single-process micro-batching: at most 16 queued/in-flight images, "
            "batches of up to 4, idle model waits up to 50 ms to fill a batch, "
            "one batch infers at a time. Cancelled/expired orders never yield "
            "results. A failed batch isolates the model (503 for new work), "
            "fails the batch and the queue, and requires a successful recovery "
            "probe before new work is accepted."
        ),
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = settings

    @app.post("/predict", status_code=201)
    async def predict(file: UploadFile = File(...), deadline_ms: int = Form(...)):
        """Submit an image with a deadline in milliseconds counted from acceptance."""
        if engine.is_isolated():
            # Reject before touching the upload: nothing new may enter while
            # the model is suspect.
            raise HTTPException(503, ISOLATED_DETAIL)
        if deadline_ms <= 0:
            raise HTTPException(400, "deadline_ms must be a positive integer")
        data = await file.read()
        try:
            tensor = preprocess(data)
        except ValueError as exc:
            raise HTTPException(400, f"invalid image: {exc}") from exc
        order, reason = engine.submit(tensor, deadline_ms)
        if order is None:
            if reason == "isolated":
                raise HTTPException(503, ISOLATED_DETAIL)
            raise HTTPException(
                429,
                f"capacity full: at most {engine.max_capacity} images queued or in-flight",
            )
        return order_view(order)

    @app.get("/orders/{order_id}")
    async def get_order(order_id: str):
        order = engine.get(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        return order_view(order)

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(order_id: str):
        order = engine.cancel(order_id)
        if order is None:
            raise HTTPException(404, "unknown order_id")
        if order.state == COMPLETED:
            raise HTTPException(409, "order already completed; cannot cancel")
        return order_view(order)

    @app.post("/recovery/probe")
    async def recovery_probe():
        """Run (or retry) the one recovery probe. 200 = recovered, 503 = still isolated."""
        result = await engine.probe()
        body = {
            "recovered": result.ok,
            "isolated": engine.is_isolated(),
            "probe": result.as_dict(),
        }
        if result.reason == "already_running":
            body["detail"] = "a recovery probe is already in progress; retry later"
            return JSONResponse(status_code=409, content=body)
        return JSONResponse(status_code=200 if result.ok else 503, content=body)

    @app.get("/recovery")
    async def recovery_status():
        last = engine.last_probe
        return {
            "isolated": engine.is_isolated(),
            "accepting_orders": not engine.is_isolated(),
            "probe_running": engine.probe_running(),
            "last_probe": last.as_dict() if last else None,
        }

    @app.get("/stats")
    async def stats():
        return engine.stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
