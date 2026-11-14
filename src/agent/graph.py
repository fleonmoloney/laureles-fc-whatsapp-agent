"""Grafo de LangGraph del agente.

    START ─► check_cache ─┬─ hit ───────────────────────────────────► END
                          └─ miss ─► retrieve ─► generate ─► evaluate_confidence ─┬─► respond  ─► END
                                                                                  └─► escalate ─► END

Supervisión humana: si no hay respuesta con sustento (el generador se abstuvo) o la confianza
queda por debajo del umbral, el agente NO responde la consulta: la marca `requires_human`
y la capa de transporte (src/main.py) la deriva a una persona.

La "confianza" es la similitud coseno de la mejor entrada del FAQ con el mensaje (0 si el
generador se abstuvo). No es una probabilidad calibrada: el umbral se elige midiendo con
`python -m scripts.evaluate` sobre mensajes reales etiquetados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.cache.semantic_cache import SemanticCache
from src.generation import Generator
from src.privacy import looks_personal
from src.retrieval import FaqRetriever, Hit


class AgentState(TypedDict, total=False):
    message: str
    hits: list[Hit]
    reply: str | None
    confidence: float
    cacheable: bool
    requires_human: bool
    source: str  # "cache" | "faq" | "human"


@dataclass
class AgentDeps:
    cache: SemanticCache
    retriever: FaqRetriever
    generator: Generator
    confidence_threshold: float = 0.8
    top_k: int = 3


@dataclass(frozen=True)
class AgentResult:
    reply: str | None
    requires_human: bool
    source: str
    confidence: float


def build_agent_graph(deps: AgentDeps) -> Any:
    def check_cache(state: AgentState) -> AgentState:
        message = state["message"]
        # Mensajes con datos personales no pasan por el caché: la respuesta podría ser
        # específica de la persona y no debe servirse a otra.
        cacheable = not looks_personal(message)
        if cacheable:
            hit = deps.cache.query(message)
            if hit is not None:
                return {
                    "cacheable": True,
                    "reply": hit.answer,
                    "confidence": hit.similarity,
                    "requires_human": False,
                    "source": "cache",
                }
        return {"cacheable": cacheable}

    def after_cache(state: AgentState) -> str:
        return "done" if state.get("source") == "cache" else "retrieve"

    def retrieve(state: AgentState) -> AgentState:
        return {"hits": deps.retriever.search(state["message"], deps.top_k)}

    def generate(state: AgentState) -> AgentState:
        hits = state.get("hits", [])
        reply = deps.generator.generate(state["message"], hits)
        confidence = hits[0].score if (reply and hits) else 0.0
        return {"reply": reply, "confidence": confidence}

    def evaluate_confidence(state: AgentState) -> AgentState:
        """Política de supervisión humana: sin respuesta con sustento o con confianza
        por debajo del umbral, no se responde al socio."""
        needs_human = state.get("reply") is None or state.get("confidence", 0.0) < deps.confidence_threshold
        return {"requires_human": needs_human}

    def route(state: AgentState) -> str:
        return "escalate" if state["requires_human"] else "respond"

    def respond(state: AgentState) -> AgentState:
        if state.get("cacheable") and state.get("reply"):
            # Solo se guardan respuestas que pasaron el umbral de confianza.
            deps.cache.store(state["message"], state["reply"])
        return {"source": "faq"}

    def escalate(state: AgentState) -> AgentState:
        return {"reply": None, "source": "human"}

    g = StateGraph(AgentState)
    g.add_node("check_cache", check_cache)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("evaluate_confidence", evaluate_confidence)
    g.add_node("respond", respond)
    g.add_node("escalate", escalate)

    g.add_edge(START, "check_cache")
    g.add_conditional_edges("check_cache", after_cache, {"done": END, "retrieve": "retrieve"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate_confidence")
    g.add_conditional_edges("evaluate_confidence", route, {"respond": "respond", "escalate": "escalate"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)
    return g.compile()


def answer(graph: Any, message: str) -> AgentResult:
    state = graph.invoke({"message": message})
    return AgentResult(
        reply=state.get("reply"),
        requires_human=bool(state.get("requires_human")),
        source=state.get("source", "human"),
        confidence=float(state.get("confidence", 0.0)),
    )
