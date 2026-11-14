"""Mide con datos etiquetados dos decisiones que el README de la primera versión daba por hechas.

1. Umbral de confianza: ¿cuántos mensajes se responden bien, cuántos se responden MAL y cuántos
   se derivan a una persona, para cada umbral?
2. Umbral del caché semántico: ¿cuántos hits son correctos y cuántos sirven una respuesta ajena?

    python -m scripts.evaluate --out results/eval_tfidf.md

Con --embedder minilm usa embeddings semánticos (requiere requirements-minilm.txt y descarga el modelo).
Las cifras salen del set de `data/eval.example.jsonl`, escrito por el autor del código: sirven para
probar el método, NO para elegir umbrales de producción. Hay que repetirlo con mensajes reales.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.cache.semantic_cache import SemanticCache
from src.embeddings import build_embedder
from src.retrieval import FaqRetriever, faq_corpus, load_faq

ROUTING_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CACHE_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}; {hi:.2f}]" if n else "0/0"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def routing_table(retriever: FaqRetriever, rows: list[dict]) -> list[str]:
    tops = [(r["expect"], retriever.search(r["message"], 1)[0]) for r in rows]
    n = len(tops)
    lines = [
        "| Umbral | Respuesta correcta | Respuesta equivocada | Derivado (debía responderse) | Derivado (correcto) |",
        "|---|---|---|---|---|",
    ]
    for t in ROUTING_THRESHOLDS:
        ok = wrong = false_esc = right_esc = 0
        for expect, top in tops:
            answers = top.score >= t
            if answers and expect is not None and top.entry_id == expect:
                ok += 1
            elif answers:
                wrong += 1  # contestó algo equivocado, o contestó algo que debía derivarse
            elif expect is not None:
                false_esc += 1
            else:
                right_esc += 1
        lines.append(
            f"| {t:.1f} | {fmt(ok, n)} | {fmt(wrong, n)} | {fmt(false_esc, n)} | {fmt(right_esc, n)} |"
        )
    return lines


def cache_table(embedder, entries, rows: list[dict]) -> list[str]:
    by_id = {e.id: e for e in entries}
    n = len(rows)
    lines = [
        "| Umbral caché | Hit correcto | Hit con respuesta ajena | Sin hit |",
        "|---|---|---|---|",
    ]
    for t in CACHE_THRESHOLDS:
        cache = SemanticCache(embedder, threshold=t)
        for e in entries:  # como si cada pregunta canónica ya hubiera sido respondida una vez
            cache.store(e.question, e.answer)
        good = bad = miss = 0
        for r in rows:
            hit = cache.query(r["message"])
            if hit is None:
                miss += 1
            elif r["expect"] is not None and hit.answer == by_id[r["expect"]].answer:
                good += 1
            else:
                bad += 1
        lines.append(f"| {t:.2f} | {fmt(good, n)} | {fmt(bad, n)} | {fmt(miss, n)} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faq", default="data/faq.example.json")
    ap.add_argument("--evalset", default="data/eval.example.jsonl")
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "minilm"])
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--out", default=None, help="archivo .md donde guardar el informe")
    args = ap.parse_args()

    entries = load_faq(args.faq)
    rows = load_rows(Path(args.evalset))
    embedder = build_embedder(args.embedder, faq_corpus(entries), args.model)
    retriever = FaqRetriever(entries, embedder)

    n_out = sum(1 for r in rows if r["expect"] is None)
    report = [
        f"# Evaluación ({args.embedder})",
        "",
        f"FAQ: `{args.faq}` ({len(entries)} entradas). Mensajes: `{args.evalset}` "
        f"({len(rows)}; {len(rows) - n_out} con respuesta en el FAQ y {n_out} que deben derivarse). "
        "Intervalos de Wilson al 95 % entre corchetes.",
        "",
        "**El set lo escribió el autor del código: es optimista y chico. Sirve para probar el método, "
        "no para fijar umbrales de producción.**",
        "",
        "## Umbral de confianza (derivación a una persona)",
        "",
        *routing_table(retriever, rows),
        "",
        "## Umbral del caché semántico",
        "",
        "Se precarga el caché con la pregunta canónica de cada entrada y se consulta con cada mensaje.",
        "",
        *cache_table(embedder, entries, rows),
        "",
    ]
    text = "\n".join(report)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
