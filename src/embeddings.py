"""Embedders intercambiables. Todos devuelven filas normalizadas (norma L2 = 1 o 0),
así que el coseno es un producto punto."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Línea base léxica: TF-IDF (1-2 gramas, sin tildes) ajustado sobre un corpus de referencia.

    No descarga nada y es determinista, pero NO entiende sinónimos: "cuánto sale" y "cuál es el
    precio" apenas se parecen. Para paraguas semántico usar MiniLMEmbedder.
    """

    def __init__(self, corpus: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            strip_accents="unicode", lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        )
        self._vec.fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray()


class MiniLMEmbedder:
    """Embeddings semánticos con sentence-transformers (descarga el modelo la primera vez)."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


def build_embedder(kind: str, corpus: list[str], model_name: str) -> Embedder:
    if kind == "tfidf":
        return TfidfEmbedder(corpus)
    if kind == "minilm":
        return MiniLMEmbedder(model_name)
    raise ValueError(f"EMBEDDER desconocido: {kind!r} (usar 'tfidf' o 'minilm')")
