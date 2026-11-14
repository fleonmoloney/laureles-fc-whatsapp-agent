from src.agent.graph import AgentDeps, answer, build_agent_graph
from src.cache.semantic_cache import SemanticCache
from src.embeddings import TfidfEmbedder
from src.retrieval import Hit

CORPUS = ["cuánto sale la cuota", "a qué hora abre la secretaría", "cuánto cuesta la entrada"]


class StubRetriever:
    def __init__(self, score):
        self.score = score

    def search(self, query, k=3):
        return [Hit("e1", "t", "pregunta", "respuesta del FAQ", self.score)]


class StubGenerator:
    def __init__(self, reply="respuesta del FAQ"):
        self.reply = reply
        self.calls = 0

    def generate(self, question, hits):
        self.calls += 1
        return self.reply


def build(score=0.9, reply="respuesta del FAQ", threshold=0.8):
    gen = StubGenerator(reply)
    cache = SemanticCache(TfidfEmbedder(CORPUS), threshold=0.95)
    graph = build_agent_graph(AgentDeps(cache, StubRetriever(score), gen, confidence_threshold=threshold))
    return graph, gen, cache


def test_confident_answer_is_returned_and_cached():
    graph, gen, cache = build(score=0.9)
    r = answer(graph, "cuánto sale la cuota")
    assert (r.reply, r.requires_human, r.source) == ("respuesta del FAQ", False, "faq")
    assert cache.stats()["entries"] == 1


def test_second_identical_question_skips_the_generator():
    graph, gen, _ = build(score=0.9)
    answer(graph, "cuánto sale la cuota")
    r = answer(graph, "cuánto sale la cuota")
    assert r.source == "cache" and r.reply == "respuesta del FAQ"
    assert gen.calls == 1  # la segunda vez no se llamó al LLM


def test_low_confidence_is_escalated_and_not_answered_or_cached():
    graph, _, cache = build(score=0.5)
    r = answer(graph, "cuánto sale la cuota")
    assert (r.reply, r.requires_human, r.source) == (None, True, "human")
    assert cache.stats()["entries"] == 0


def test_generator_abstention_escalates_even_with_a_high_score():
    graph, _, _ = build(score=0.99, reply=None)
    r = answer(graph, "cuánto sale la cuota")
    assert r.requires_human and r.reply is None and r.confidence == 0.0


def test_score_equal_to_threshold_is_answered():
    graph, _, _ = build(score=0.8, threshold=0.8)
    assert not answer(graph, "cuánto sale la cuota").requires_human


def test_messages_with_personal_data_are_never_cached():
    graph, gen, cache = build(score=0.95)
    msg = "mi cédula es 1.234.567-8, cuánto sale la cuota"
    answer(graph, msg)
    answer(graph, msg)
    assert cache.stats()["entries"] == 0
    assert gen.calls == 2  # se respondió cada vez, sin reutilizar nada


def test_escalated_messages_do_not_poison_the_cache_for_later_ones():
    graph, _, cache = build(score=0.3)
    answer(graph, "cuánto sale la cuota")
    answer(graph, "cuánto sale la cuota")
    assert cache.stats()["hits"] == 0
