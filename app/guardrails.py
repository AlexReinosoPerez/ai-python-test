"""
Guardrails module — robust parsing pipeline for noisy AI responses.

The mock AI provider returns responses with the following distribution:
  50%  Clean JSON
  10%  Variant keys (Recipient/body/channel, To/Message/Type, destination/text/method)
  10%  Extra or missing fields
  10%  JSON wrapped in markdown code blocks
  10%  Malformed JSON (truncated, single quotes, unquoted keys)
  10%  Refusal / error messages

This module implements a multi-stage pipeline to handle all cases.
"""

import re
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger("app.guardrails")

# ─── Key normalisation map ────────────────────────────────────────────────────
# Maps every known variant key (lowercased) to the canonical field name.
KEY_ALIASES: Dict[str, str] = {
    # canonical
    "to": "to",
    "message": "message",
    "type": "type",
    # variant set 1
    "recipient": "to",
    "body": "message",
    "channel": "type",
    # variant set 2
    "destination": "to",
    "text": "message",
    "method": "type",
}

# ─── Refusal detection ────────────────────────────────────────────────────────
REFUSAL_PATTERNS = [
    "no tengo permitido",
    "viola las políticas",
    "content analysis flagged",
    "refused:",
    "cannot process",
    "no puedo procesar",
    "unable to identify",
    "i was unable",
    "i cannot process",
]


def is_refusal(content: str) -> bool:
    """Return True if the AI response is a refusal or policy-based rejection."""
    lower = content.lower()
    return any(pat in lower for pat in REFUSAL_PATTERNS)


# ─── Stage 1: Strip markdown wrappers ─────────────────────────────────────────

def strip_markdown_blocks(content: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        logger.debug("Stage 1 — Stripped markdown wrapper → '%.200s'", extracted)
        return extracted
    return content


# ─── Stage 2: Extract JSON substring from mixed text ──────────────────────────

def extract_json_substring(text: str) -> Optional[str]:
    """Find the first top-level {...} in arbitrary prose."""
    depth = 0
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                result = text[start : i + 1]
                logger.debug("Stage 2 — Extracted JSON substring (len=%d)", len(result))
                return result
    # unclosed brace – return partial so fix_malformed can close it
    if start is not None:
        partial = text[start:]
        logger.debug("Stage 2 — Found unclosed JSON (partial len=%d)", len(partial))
        return partial
    return None


# ─── Stage 3: Fix common malformations ────────────────────────────────────────

def fix_malformed_json(text: str) -> str:
    """Best-effort repair of broken JSON strings."""
    s = text.strip()
    repairs: list[str] = []

    # Remove trailing ellipsis (truncated responses)
    cleaned = re.sub(r"\s*\.{2,}\s*$", "", s)
    if cleaned != s:
        repairs.append("removed trailing ellipsis")
        s = cleaned

    # Single quotes → double quotes
    if "'" in s and s.lstrip().startswith("{"):
        s = s.replace("'", '"')
        repairs.append("single→double quotes")

    # Unquoted keys:  {key: "val"} → {"key": "val"}
    new_s = re.sub(r"(?<=[{,])\s*(\w+)\s*:", r' "\1":', s)
    if new_s != s:
        repairs.append("quoted unquoted keys")
        s = new_s

    # Close unclosed braces
    diff = s.count("{") - s.count("}")
    if diff > 0:
        s += "}" * diff
        repairs.append(f"closed {diff} unclosed brace(s)")

    if repairs:
        logger.debug("Stage 3 — Repairs applied: %s", ", ".join(repairs))

    return s


# ─── Stage 4: Key normalisation ───────────────────────────────────────────────

def normalize_keys(data: dict) -> dict:
    """Map variant key names to the canonical {to, message, type} schema."""
    result: dict = {}
    mapped: list[str] = []

    for key, value in data.items():
        canonical = KEY_ALIASES.get(key.lower().strip())
        if canonical:
            result[canonical] = value
            if key.lower().strip() != canonical:
                mapped.append(f"'{key}'→'{canonical}'")
        else:
            logger.debug("Stage 4 — Discarded unknown key '%s'", key)

    if mapped:
        logger.debug("Stage 4 — Key normalisations: %s", ", ".join(mapped))

    return result


# ─── Stage 5: Infer missing fields from user input ────────────────────────────

_EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.\w+")
_PHONE_RE = re.compile(r"\b\d{3}-?\d{3}-?\d{3,4}\b")


def infer_missing_fields(data: dict, user_input: str) -> dict:
    """
    Attempt to fill 'to' and 'type' from the original natural-language prompt.

    Returns a *new* dict (does not mutate input).
    """
    result = dict(data)  # shallow copy — avoid mutating caller's dict
    lower = user_input.lower()
    inferred: list[str] = []

    # ── infer type ──
    if "type" not in result or result.get("type") not in ("email", "sms"):
        if any(kw in lower for kw in ("email", "correo", "mail")):
            result["type"] = "email"
            inferred.append("type=email (from keywords)")
        elif any(kw in lower for kw in ("sms", "teléfono", "telefono")):
            result["type"] = "sms"
            inferred.append("type=sms (from keywords)")
        elif _EMAIL_RE.search(user_input):
            result["type"] = "email"
            inferred.append("type=email (email detected in input)")
        elif _PHONE_RE.search(user_input):
            result["type"] = "sms"
            inferred.append("type=sms (phone detected in input)")

    # ── infer destination ──
    if not result.get("to"):
        email_m = _EMAIL_RE.search(user_input)
        phone_m = _PHONE_RE.search(user_input)
        if email_m:
            result["to"] = email_m.group(0)
            inferred.append(f"to={email_m.group(0)} (email from input)")
        elif phone_m:
            result["to"] = phone_m.group(0)
            inferred.append(f"to={phone_m.group(0)} (phone from input)")

    if inferred:
        logger.info("Stage 5 — Inferred missing fields: %s", ", ".join(inferred))

    return result


# ─── Public entry point ───────────────────────────────────────────────────────

def parse_ai_response(content: str, user_input: str = "") -> Optional[dict]:
    """
    Main guardrail pipeline.

    Returns a dict with {to, message, type} on success, or None on failure.
    """
    if not content or not content.strip():
        logger.warning("Guardrails received empty AI content")
        return None

    logger.debug("Guardrails input (first 200 chars): %.200s", content)

    # 0. Refusal / policy rejection
    if is_refusal(content):
        logger.warning("AI response detected as REFUSAL: '%.120s'", content)
        return None

    # 1. Strip markdown wrappers
    cleaned = strip_markdown_blocks(content)

    # 2. Try direct parse
    parsed = _try_parse(cleaned)
    if parsed is not None:
        logger.debug("Direct parse succeeded (stage 2)")
        return _finalize(parsed, user_input)

    # 3. Extract JSON from mixed text
    json_str = extract_json_substring(cleaned)
    if json_str:
        parsed = _try_parse(json_str)
        if parsed is not None:
            logger.debug("JSON substring parse succeeded (stage 3a)")
            return _finalize(parsed, user_input)

        # 4. Repair malformed JSON
        fixed = fix_malformed_json(json_str)
        parsed = _try_parse(fixed)
        if parsed is not None:
            logger.debug("Malformed JSON repair succeeded (stage 3b)")
            return _finalize(parsed, user_input)

    # 5. Repair the entire content as last resort
    fixed = fix_malformed_json(cleaned)
    parsed = _try_parse(fixed)
    if parsed is not None:
        logger.debug("Full-content repair succeeded (stage 4 — last resort)")
        return _finalize(parsed, user_input)

    logger.warning(
        "ALL parse stages failed — content (first 200 chars): %.200s", content
    )
    return None


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _try_parse(text: str) -> Optional[dict]:
    """Attempt a strict JSON parse; return dict or None."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        logger.debug("JSON parsed but type is %s, expected dict", type(data).__name__)
        return None
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug("JSON parse failed: %s — input: '%.100s'", exc, text)
        return None


def _finalize(data: dict, user_input: str) -> dict:
    """Normalize keys and infer any missing fields."""
    normalized = normalize_keys(data)
    result = infer_missing_fields(normalized, user_input)
    logger.debug("Finalized result: %s", result)
    return result
