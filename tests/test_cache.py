import threading

from src.cache.semantic_cache import SemanticCache
from src.embeddings import TfidfEmbedder

CORPUS = [
    "cuánto sale la cuota social",
    "a qué hora abre la secretaría",
    "cuánto cuesta la entrada al partido",
    "dónde se paga la cuota",
]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(**kw):
    return SemanticCache(TfidfEmbedder(CORPUS), **kw)


def test_empty_cache_misses():
    assert make().query("cuánto sale la cuota social") is None


def test_identical_and_near_identical_messages_hit():
    c = make()
    c.store("¿Cuánto sale la cuota social?", "$1.200")
    assert c.query("¿Cuánto sale la cuota social?").answer == "$1.200"
    # mayúsculas, tildes y puntuación no importan
    assert c.query("CUANTO SALE LA CUOTA SOCIAL").answer == "$1.200"


def test_different_topic_does_not_hit():
    c = make()
    c.store("cuánto sale la cuota social", "$1.200")
    assert c.query("a qué hora abre la secretaría") is None


def test_threshold_controls_strictness():
    strict, lax = make(threshold=0.99), make(threshold=0.3)
    for c in (strict, lax):
        c.store("cuánto sale la cuota social", "$1.200")
    probe = "cuánto sale la cuota"
    assert strict.query(probe) is None
    assert lax.query(probe) is not None


def test_entries_expire_after_ttl():
    clock = Clock()
    c = make(ttl_seconds=60, clock=clock)
    c.store("cuánto sale la cuota social", "$1.200")
    clock.now = 59
    assert c.query("cuánto sale la cuota social") is not None
    clock.now = 61
    assert c.query("cuánto sale la cuota social") is None
    assert c.stats()["entries"] == 0


def test_oldest_entries_are_evicted_beyond_max():
    c = make(max_entries=2)
    c.store("cuánto sale la cuota social", "a")
    c.store("a qué hora abre la secretaría", "b")
    c.store("cuánto cuesta la entrada al partido", "c")
    assert c.stats()["entries"] == 2
    assert c.stats()["evictions"] == 1
    assert c.query("cuánto sale la cuota social") is None  # la más antigua se fue


def test_stats_count_hits_and_misses():
    c = make()
    c.store("cuánto sale la cuota social", "x")
    c.query("cuánto sale la cuota social")
    c.query("dónde se paga la cuota")
    s = c.stats()
    assert (s["hits"], s["misses"], s["hit_rate"]) == (1, 1, 0.5)


def test_concurrent_use_is_safe():
    c = make(max_entries=50)
    errors = []

    def worker(i):
        try:
            for j in range(30):
                c.store(f"cuánto sale la cuota social {i} {j}", "x")
                c.query("cuánto sale la cuota social")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert c.stats()["entries"] <= 50
