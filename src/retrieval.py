"""Recuperación sobre las preguntas frecuentes del club."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.embeddings import Embedder


@dataclass(frozen=True)
class FaqEntry:
    id: str
    topic: str
    question: str
    answer: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    entry_id: str
    topic: str
    question: str
    answer: str
    score: float


def load_faq(path: str | Path) -> list[FaqEntry]:
    """Formato: {"entries": [{"id", "topic", "question", "answer", "aliases": [...]}]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [
        FaqEntry(
            id=e["id"],
            topic=e.get("topic", ""),
            question=e["question"],
            answer=e["answer"],
            aliases=tuple(e.get("aliases", ())),
        )
        for e in raw["entries"]
    ]
    if not entries:
        raise ValueError(f"{path}: el FAQ no tiene entradas")
    return entries


def faq_corpus(entries: list[FaqEntry]) -> list[str]:
    """Todos los textos de las entradas, para ajustar embedders que lo necesiten (TF-IDF)."""
    texts: list[str] = []
    for e in entries:
        texts += [e.question, *e.aliases, e.answer]
    return texts


class FaqRetriever:
    """Cada entrada se indexa con su pregunta y sus variantes; el puntaje de la entrada es el
    máximo coseno entre el mensaje y cualquiera de esos textos."""

    def __init__(self, entries: list[FaqEntry], embedder: Embedder) -> None:
        self._entries = entries
        self._embedder = embedder
        texts: list[str] = []
        owner: list[int] = []
        for i, e in enumerate(entries):
            for t in (e.question, *e.aliases):
                texts.append(t)
                owner.append(i)
        self._owner = np.asarray(owner)
        self._matrix = embedder.encode(texts)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        sims = self._matrix @ self._embedder.encode([query])[0]
        best: dict[int, float] = {}
        for idx, s in zip(self._owner, sims):
            best[int(idx)] = max(best.get(int(idx), -1.0), float(s))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Hit(
                entry_id=self._entries[i].id,
                topic=self._entries[i].topic,
                question=self._entries[i].question,
                answer=self._entries[i].answer,
                score=score,
            )
            for i, score in ranked
        ]
