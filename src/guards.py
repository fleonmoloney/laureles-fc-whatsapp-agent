"""Protecciones de costo y duplicados para el webhook."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable


class RecentIds:
    """Recuerda los últimos N ids de mensaje. WhatsApp reintenta los webhooks que no responden
    a tiempo; sin esto, un reintento duplicaría la respuesta (y el costo)."""

    def __init__(self, capacity: int = 5000) -> None:
        self._capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        """True si ya se vio; si no, lo registra y devuelve False."""
        with self._lock:
            if key in self._seen:
                return True
            self._seen[key] = None
            while len(self._seen) > self._capacity:
                self._seen.popitem(last=False)
            return False


class SlidingWindowLimiter:
    """Máximo `limit` eventos por clave en una ventana móvil. Acota el gasto de LLM que un
    solo remitente (o un bucle) puede generar. `limit <= 0` desactiva el límite."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self._limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            q = self._events.setdefault(key, deque())
            while q and now - q[0] >= self._window:
                q.popleft()
            if len(q) >= self._limit:
                return False
            q.append(now)
            return True
