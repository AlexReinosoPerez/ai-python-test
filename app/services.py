"""
External service clients — AI extraction and notification provider.

Both endpoints live at localhost:3001 (shared Docker network).
Uses tenacity for automatic retries with exponential backoff on transient errors.
"""

import httpx
import logging
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger("app.services")

# ─── Configuration ─────────────────────────────────────────────────────────────
PROVIDER_BASE_URL = "http://localhost:3001"
API_KEY = "test-dev-2026"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

SYSTEM_PROMPT = (
    "You are a structured data extraction assistant. "
    "Analyze the user's natural language message and extract notification details. "
    'Return ONLY a valid JSON object with these exact fields:\n'
    '  "to"      – the recipient (email address or phone number)\n'
    '  "message" – the body/content of the notification\n'
    '  "type"    – either "email" or "sms"\n\n'
    "Rules:\n"
    '- If the user mentions email, correo, or mail → type = "email"\n'
    '- If the user mentions SMS, teléfono, or phone → type = "sms"\n'
    "- Extract the exact email address or phone number for the 'to' field\n"
    "- The 'message' is the content the user wants to send\n"
    "- Return ONLY the raw JSON object — no markdown, no explanation, no extra text"
)


# ─── Retry predicate ──────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429/5xx and transport-level errors; never on 401/422."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        retryable = code in (429, 500, 502, 503, 504)
        logger.debug(
            "Retry predicate: HTTP %d → %s",
            code,
            "retryable" if retryable else "NOT retryable",
        )
        return retryable
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        logger.debug("Retry predicate: %s → retryable", type(exc).__name__)
        return True
    return False


# ─── AI Extraction ─────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def call_ai_extract(user_input: str, client: httpx.AsyncClient) -> str:
    """
    Send the user's natural-language input to the AI extraction endpoint.
    Returns the raw content string from the AI response.

    Raises:
        httpx.HTTPStatusError: on non-retryable HTTP errors (after exhausting retries for retryable ones)
        httpx.TransportError / httpx.TimeoutException: on network-level failures
        ValueError: if the AI response structure is unexpected
    """
    url = f"{PROVIDER_BASE_URL}/v1/ai/extract"
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    }

    logger.debug("AI extract → POST %s (input length=%d)", url, len(user_input))
    t0 = time.monotonic()

    response = await client.post(url, json=payload, headers=HEADERS)
    response.raise_for_status()

    elapsed = time.monotonic() - t0
    data = response.json()

    # ── Defensive check on response structure ──────────────────────────────
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error(
            "AI response has unexpected structure: %s — raw: %.300s",
            exc,
            data,
        )
        raise ValueError(f"Malformed AI response: {exc}") from exc

    logger.info(
        "AI extract ← %.2fs — HTTP %d — content length=%d",
        elapsed,
        response.status_code,
        len(content),
    )
    return content


# ─── Notification Dispatch ─────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
    retry=retry_if_exception(_is_retryable),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def call_notify(
    to: str, message: str, notif_type: str, client: httpx.AsyncClient
) -> dict:
    """
    Send a notification through the provider.

    Returns the provider response dict on success.

    Raises:
        httpx.HTTPStatusError: on non-retryable HTTP errors
        httpx.TransportError / httpx.TimeoutException: on network failures
    """
    url = f"{PROVIDER_BASE_URL}/v1/notify"
    payload = {"to": to, "message": message, "type": notif_type}

    logger.debug("Notify → POST %s — to='%s' type='%s'", url, to, notif_type)
    t0 = time.monotonic()

    response = await client.post(url, json=payload, headers=HEADERS)
    response.raise_for_status()

    elapsed = time.monotonic() - t0
    result = response.json()

    logger.info(
        "Notify ← %.2fs — HTTP %d — provider_id=%s status=%s",
        elapsed,
        response.status_code,
        result.get("provider_id", "?"),
        result.get("status", "?"),
    )
    return result
