"""Retrieval, privacy, config, guards y el método de evaluación."""

import json
from pathlib import Path

import pytest

from scripts.evaluate import routing_table, wilson
from src.config import Settings
from src.embeddings import TfidfEmbedder, build_embedder
from src.guards import RecentIds, SlidingWindowLimiter
from src.privacy import looks_personal
from src.retrieval import FaqRetriever, faq_corpus, load_faq

ROOT = Path(__file__).resolve().parent.parent
FAQ = ROOT / "data" / "faq.example.json"


# --- recuperación -----------------------------------------------------------------------------------

def retriever():
    entries = load_faq(FAQ)
    return FaqRetriever(entries, TfidfEmbedder(faq_corpus(entries)))


def test_canonical_question_ranks_first_with_a_perfect_score():
    hit = retriever().search("¿Cuánto sale la cuota social?", 3)[0]
    assert hit.entry_id == "cuota_monto"
    assert hit.score == pytest.approx(1.0)


def test_alias_and_paraphrase_find_the_right_entry():
    r = retriever()
    assert r.search("quiero asociarme al club", 1)[0].entry_id == "alta_socio"
    assert r.search("perdí mi carnet de socio", 1)[0].entry_id == "carnet"


def test_search_respects_k_and_orders_by_score():
    hits = retriever().search("cuota", 3)
    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_empty_faq_is_rejected(tmp_path):
    p = tmp_path / "faq.json"
    p.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_faq(p)


def test_unknown_embedder_is_rejected():
    with pytest.raises(ValueError):
        build_embedder("magia", ["a"], "m")


# --- privacidad ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "mi cédula es 1.234.567-8", "llamame al 099 123 456", "escribime a juan@correo.com", "socio 12345678",
])
def test_personal_data_is_detected(text):
    assert looks_personal(text)


@pytest.mark.parametrize("text", ["¿cuánto sale la cuota?", "abre a las 15:30", "la cuota es $1.200", "tengo 3 hijos"])
def test_ordinary_questions_are_not_flagged(text):
    assert not looks_personal(text)


# --- configuración -------------------------------------------------------------------------------------

def test_missing_required_variables_are_all_reported():
    with pytest.raises(RuntimeError) as exc:
        Settings.from_env({})
    msg = str(exc.value)
    for name in ("WHATSAPP_VERIFY_TOKEN", "WHATSAPP_APP_SECRET", "WHATSAPP_ACCESS_TOKEN",
                 "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_API_VERSION", "FAQ_PATH"):
        assert name in msg


def test_settings_parse_overrides_and_defaults():
    env = dict(WHATSAPP_VERIFY_TOKEN="v", WHATSAPP_APP_SECRET="s", WHATSAPP_ACCESS_TOKEN="t",
               WHATSAPP_PHONE_NUMBER_ID="1", WHATSAPP_API_VERSION="vX", FAQ_PATH="f.json")
    s = Settings.from_env(env)
    assert (s.confidence_threshold, s.cache_threshold, s.rate_limit_per_minute, s.embedder) == (0.8, 0.95, 10, "tfidf")
    assert s.operator_wa_id is None and s.llm_model is None
    s2 = Settings.from_env({**env, "CONFIDENCE_THRESHOLD": "0.6", "RATE_LIMIT_PER_MINUTE": "3", "OPERATOR_WA_ID": "598"})
    assert (s2.confidence_threshold, s2.rate_limit_per_minute, s2.operator_wa_id) == (0.6, 3, "598")


# --- protecciones -------------------------------------------------------------------------------------

def test_recent_ids_detects_duplicates_and_forgets_the_oldest():
    r = RecentIds(capacity=2)
    assert not r.seen("a") and r.seen("a")
    r.seen("b")
    r.seen("c")  # desplaza a "a"
    assert not r.seen("a")


def test_limiter_uses_a_sliding_window_per_key():
    now = [0.0]
    lim = SlidingWindowLimiter(2, 60, clock=lambda: now[0])
    assert lim.allow("x") and lim.allow("x") and not lim.allow("x")
    assert lim.allow("y")  # otra clave no se ve afectada
    now[0] = 61
    assert lim.allow("x")


def test_limiter_zero_disables_the_limit():
    lim = SlidingWindowLimiter(0)
    assert all(lim.allow("x") for _ in range(100))


# --- método de evaluación -------------------------------------------------------------------------------

def test_wilson_interval_known_values():
    lo, hi = wilson(0, 35)
    assert lo == 0.0 and hi == pytest.approx(0.0987, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_routing_table_counts_add_up():
    r = retriever()
    rows = [{"message": "¿Cuánto sale la cuota social?", "expect": "cuota_monto"},
            {"message": "xyzzy qwerty", "expect": None}]
    lines = routing_table(r, rows)
    assert len(lines) == 2 + 7  # encabezado + 7 umbrales
    assert "| 0.9 | 1/2" in lines[-1]  # la pregunta canónica se responde bien incluso con umbral alto
