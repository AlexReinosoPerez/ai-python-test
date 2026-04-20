"""In-memory request store for tracking notification lifecycle."""

import logging
from typing import Dict, Literal, Optional

from models import RequestRecord

logger = logging.getLogger("app.store")

# Valid status transitions — enforced at runtime to catch programming errors.
_VALID_STATUSES = frozenset({"queued", "processing", "sent", "failed"})

StatusLiteral = Literal["queued", "processing", "sent", "failed"]


class RequestStore:
    """
    Simple in-memory store.

    Thread-safety note: all mutations are synchronous and execute
    inside a single asyncio event loop, so dict operations are
    atomic under CPython's GIL.
    """

    def __init__(self) -> None:
        self._data: Dict[str, RequestRecord] = {}

    # ── write ──────────────────────────────────────────────────────────────

    def save(self, record: RequestRecord) -> None:
        """Persist a new request record."""
        self._data[record.id] = record
        logger.debug("[%s] Saved — status=%s", record.id, record.status)

    def update_status(self, request_id: str, status: StatusLiteral) -> bool:
        """
        Transition the record to *status*.

        Returns True on success, False if the id doesn't exist.
        Raises ValueError if *status* is not in the allowed set.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of {_VALID_STATUSES}"
            )

        record = self._data.get(request_id)
        if record is None:
            logger.warning("[%s] update_status('%s') — record not found", request_id, status)
            return False

        old = record.status
        record.status = status
        logger.debug("[%s] Status transition: %s → %s", request_id, old, status)
        return True

    # ── read ───────────────────────────────────────────────────────────────

    def get(self, request_id: str) -> Optional[RequestRecord]:
        """Return the record or None."""
        record = self._data.get(request_id)
        if record is None:
            logger.debug("[%s] get() — not found (store size=%d)", request_id, len(self._data))
        return record

    @property
    def size(self) -> int:
        """Current number of stored records (useful for monitoring)."""
        return len(self._data)


# Singleton instance used across the application
store = RequestStore()
