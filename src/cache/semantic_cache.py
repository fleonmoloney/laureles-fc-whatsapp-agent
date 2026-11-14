"""Caché semántico: si un mensaje nuevo es casi idéntico en significado a uno ya respondido
(coseno >= umbral), se reutiliza la respuesta sin llamar al LLM.

Diferencias respecto del piloto original: vence por tiempo (los horarios y fixtures cambian),
tiene tamaño máximo, expone estadísticas, es seguro entre hilos y no imprime en consola.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.embeddings import Embedder


@dataclass(frozen=True)
class CacheHit:
    answer: str
    similarity: float


@dataclass
class _Entry:
    question: str
    answer: str
    embedding: np.ndarray
    created: float


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.95,
        ttl_seconds: float = 6 * 3600,
        max_entries: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._embedder = embedder
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _purge_expired(self) -> None:
        now = self._clock()
        before = len(self._entries)
        self._entries = [e for e in self._entries if now - e.created < self.ttl_seconds]
        self._evictions += before - len(self._entries)

    def query(self, message: str) -> CacheHit | None:
        # La codificación (la parte lenta) va fuera del candado.
        q = self._embedder.encode([message])[0]
        with self._lock:
            self._purge_expired()
            if not self._entries:
                self._misses += 1
                return None
            matrix = np.stack([e.embedding for e in self._entries])
            sims = matrix @ q
            best = int(np.argmax(sims))
            if float(sims[best]) >= self.threshold:
                self._hits += 1
                return CacheHit(self._entries[best].answer, float(sims[best]))
            self._misses += 1
            return None

    def store(self, question: str, answer: str) -> None:
        emb = self._embedder.encode([question])[0]
        with self._lock:
            self._purge_expired()
            self._entries.append(_Entry(question, answer, emb, self._clock()))
            overflow = len(self._entries) - self.max_entries
            if overflow > 0:
                del self._entries[:overflow]  # descarta las más antiguas
                self._evictions += overflow

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, float]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "entries": len(self._entries),
                "evictions": self._evictions,
            }
