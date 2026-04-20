"""
Intelligent Notification Service — FastAPI application.

Pipeline:
  1. POST /v1/requests              → Ingest natural-language request
  2. POST /v1/requests/{id}/process → AI extraction → guardrails → notify
  3. GET  /v1/requests/{id}         → Query current status
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import httpx
import time
import uuid
import logging

from models import RequestInput, RequestRecord, ExtractedData
from store import store
from guardrails import parse_ai_response
from services import call_ai_extract, call_notify

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("app.main")

# ─── Shared HTTP client (connection-pooled) ────────────────────────────────────
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the shared async HTTP client."""
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    logger.info("Lifespan started — HTTP client initialised (timeout=30s)")
    yield
    await http_client.aclose()
    logger.info("Lifespan ended — HTTP client closed gracefully")


app = FastAPI(
    title="Notification Service (Technical Test)",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 1 — Ingest
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/requests", status_code=201)
async def create_request(body: RequestInput):
    """Accept a natural-language notification request and return a tracking id."""
    request_id = str(uuid.uuid4())
    record = RequestRecord(id=request_id, user_input=body.user_input)
    store.save(record)
    logger.info(
        "[%s] Request created — user_input='%.120s'",
        request_id,
        body.user_input,
    )
    return {"id": request_id}


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 2 — Process (AI extraction → guardrails → notification)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/requests/{request_id}/process")
async def process_request(request_id: str):
    """
    Orchestrate the full processing pipeline for a queued request:
      1. Call AI extraction endpoint
      2. Parse & sanitise the response through guardrails
      3. Validate structured output with Pydantic
      4. Dispatch notification to the provider
    """
    record = store.get(request_id)
    if not record:
        logger.warning("[%s] Process requested but ID not found in store", request_id)
        raise HTTPException(status_code=404, detail="Request not found")

    logger.info("[%s] ── Pipeline START ── status=%s", request_id, record.status)
    store.update_status(request_id, "processing")
    pipeline_start = time.monotonic()

    try:
        # ── Step 1: AI extraction ──────────────────────────────────────────
        logger.info("[%s] Step 1/4 — Calling AI extraction…", request_id)
        ai_start = time.monotonic()

        try:
            ai_content = await call_ai_extract(record.user_input, http_client)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[%s] AI extraction HTTP error: status=%d body=%.200s",
                request_id,
                exc.response.status_code,
                exc.response.text,
            )
            store.update_status(request_id, "failed")
            return {"id": request_id, "status": "failed"}
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            logger.error(
                "[%s] AI extraction transport/timeout error: %s",
                request_id,
                exc,
                exc_info=True,
            )
            store.update_status(request_id, "failed")
            return {"id": request_id, "status": "failed"}

        ai_elapsed = time.monotonic() - ai_start
        logger.info(
            "[%s] Step 1/4 — AI responded in %.2fs — raw (first 300 chars): %.300s",
            request_id,
            ai_elapsed,
            ai_content,
        )

        # ── Step 2: Guardrails — parse, clean, normalise ──────────────────
        logger.info("[%s] Step 2/4 — Running guardrails pipeline…", request_id)
        parsed = parse_ai_response(ai_content, record.user_input)

        if parsed is None:
            store.update_status(request_id, "failed")
            logger.warning(
                "[%s] Step 2/4 — Guardrails FAILED — could not extract structured data",
                request_id,
            )
            return {"id": request_id, "status": "failed"}

        logger.info("[%s] Step 2/4 — Guardrails OK — parsed=%s", request_id, parsed)

        # ── Step 3: Pydantic validation ────────────────────────────────────
        logger.info("[%s] Step 3/4 — Validating with Pydantic…", request_id)
        try:
            extracted = ExtractedData(**parsed)
        except Exception as exc:
            store.update_status(request_id, "failed")
            logger.warning(
                "[%s] Step 3/4 — Pydantic validation FAILED — data=%s error=%s",
                request_id,
                parsed,
                exc,
            )
            return {"id": request_id, "status": "failed"}

        logger.info(
            "[%s] Step 3/4 — Validation OK — to='%s' type='%s' message='%.80s'",
            request_id,
            extracted.to,
            extracted.type,
            extracted.message,
        )

        # ── Step 4: Send notification ──────────────────────────────────────
        logger.info("[%s] Step 4/4 — Sending notification…", request_id)
        notify_start = time.monotonic()

        try:
            result = await call_notify(
                extracted.to, extracted.message, extracted.type, http_client
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[%s] Notification HTTP error: status=%d body=%.200s",
                request_id,
                exc.response.status_code,
                exc.response.text,
            )
            store.update_status(request_id, "failed")
            return {"id": request_id, "status": "failed"}
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            logger.error(
                "[%s] Notification transport/timeout error: %s",
                request_id,
                exc,
                exc_info=True,
            )
            store.update_status(request_id, "failed")
            return {"id": request_id, "status": "failed"}

        notify_elapsed = time.monotonic() - notify_start
        provider_id = result.get("provider_id", "unknown")
        logger.info(
            "[%s] Step 4/4 — Notification SENT in %.2fs — provider_id=%s",
            request_id,
            notify_elapsed,
            provider_id,
        )

        store.update_status(request_id, "sent")
        total = time.monotonic() - pipeline_start
        logger.info(
            "[%s] ── Pipeline COMPLETE ── total=%.2fs (ai=%.2fs, notify=%.2fs)",
            request_id,
            total,
            ai_elapsed,
            notify_elapsed,
        )
        return {"id": request_id, "status": "sent"}

    except Exception as exc:
        store.update_status(request_id, "failed")
        total = time.monotonic() - pipeline_start
        logger.error(
            "[%s] ── Pipeline UNEXPECTED ERROR after %.2fs ── %s: %s",
            request_id,
            total,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return {"id": request_id, "status": "failed"}


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint 3 — Status query
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/requests/{request_id}")
async def get_request_status(request_id: str):
    """Return the current status of a notification request."""
    record = store.get(request_id)
    if not record:
        logger.debug("[%s] Status query — not found", request_id)
        raise HTTPException(status_code=404, detail="Request not found")
    logger.debug("[%s] Status query — status=%s", request_id, record.status)
    return {"id": record.id, "status": record.status}
