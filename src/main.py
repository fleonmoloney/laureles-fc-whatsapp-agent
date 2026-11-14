"""Servicio FastAPI: recibe webhooks de WhatsApp, corre el agente y responde o deriva a una persona.

    uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000

    GET  /webhook   verificación de Meta (hub.mode / hub.verify_token / hub.challenge)
    POST /webhook   mensajes entrantes; firma X-Hub-Signature-256 obligatoria; responde 200 y procesa aparte
    GET  /health
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from src.agent.graph import AgentDeps, answer, build_agent_graph
from src.cache.semantic_cache import SemanticCache
from src.config import Settings
from src.embeddings import build_embedder
from src.generation import AnthropicGenerator, ExtractiveGenerator, Generator
from src.guards import RecentIds, SlidingWindowLimiter
from src.retrieval import FaqRetriever, faq_corpus, load_faq
from src.whatsapp import IncomingMessage, WhatsAppClient, parse_messages, verify_signature

log = logging.getLogger("laureles.agent")

MAX_INPUT_CHARS = 1000  # tope de lo que se envía al recuperador / LLM
NON_TEXT_REPLY = "Por ahora solo puedo leer mensajes de texto. ¿Podés escribirme tu consulta?"


@dataclass(frozen=True)
class Handoff:
    created: float
    sender: str  # hash corto, nunca el número
    confidence: float


def _anon(wa_id: str) -> str:
    return hashlib.sha256(wa_id.encode()).hexdigest()[:8]


def _default_generator(settings: Settings) -> Generator:
    if settings.llm_model and os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        return AnthropicGenerator(anthropic.Anthropic(), settings.llm_model)
    return ExtractiveGenerator()


def create_app(
    settings: Settings | None = None,
    *,
    wa_client: WhatsAppClient | None = None,
    generator: Generator | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.embedder == "tfidf":
        log.warning(
            "EMBEDDER=tfidf: línea base léxica sin descargas. Los umbrales por defecto (confianza 0.8, "
            "caché 0.95) están pensados para embeddings semánticos (EMBEDDER=minilm): con TF-IDF casi "
            "todo se deriva a una persona y el caché solo acierta con mensajes casi idénticos. "
            "Calibrar con `python -m scripts.evaluate`."
        )

    entries = load_faq(settings.faq_path)
    embedder = build_embedder(settings.embedder, faq_corpus(entries), settings.embedder_model)
    cache = SemanticCache(
        embedder,
        threshold=settings.cache_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
        max_entries=settings.cache_max_entries,
    )
    graph = build_agent_graph(
        AgentDeps(
            cache=cache,
            retriever=FaqRetriever(entries, embedder),
            generator=generator or _default_generator(settings),
            confidence_threshold=settings.confidence_threshold,
        )
    )
    wa = wa_client or WhatsAppClient(
        settings.access_token, settings.phone_number_id, settings.api_version
    )
    dedup = RecentIds()
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    handoffs: deque[Handoff] = deque(maxlen=200)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        wa.close()

    app = FastAPI(title="Laureles F.C. - agente de WhatsApp", lifespan=lifespan)
    app.state.handoffs = handoffs
    app.state.cache = cache

    def process(msg: IncomingMessage) -> None:
        who = _anon(msg.wa_id)
        if not limiter.allow(msg.wa_id):
            log.warning("límite de mensajes superado sender=%s", who)
            return
        try:
            if msg.text is None:
                wa.send_text(msg.wa_id, NON_TEXT_REPLY)
                return
            result = answer(graph, msg.text[:MAX_INPUT_CHARS])
            log.info("procesado sender=%s source=%s confidence=%.2f", who, result.source, result.confidence)
            if result.requires_human:
                handoffs.append(Handoff(time.time(), who, result.confidence))
                if settings.handoff_ack:
                    wa.send_text(msg.wa_id, settings.handoff_ack)
                if settings.operator_wa_id:
                    wa.send_text(
                        settings.operator_wa_id,
                        f"Consulta de {msg.wa_id} para atender: {msg.text[:500]}",
                    )
            elif result.reply:
                wa.send_text(msg.wa_id, result.reply)
        except Exception as exc:  # un mensaje defectuoso no debe tumbar el servicio
            log.error("falló el procesamiento sender=%s error=%s", who, type(exc).__name__)

    @app.get("/webhook")
    def verify(
        hub_mode: str | None = Query(default=None, alias="hub.mode"),
        hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
        hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
    ) -> PlainTextResponse:
        ok = (
            hub_mode == "subscribe"
            and hub_challenge is not None
            and hub_verify_token is not None
            and hmac.compare_digest(hub_verify_token.encode(), settings.verify_token.encode())
        )
        if not ok:
            raise HTTPException(status_code=403, detail="verificación rechazada")
        return PlainTextResponse(hub_challenge)

    @app.post("/webhook")
    async def receive(
        request: Request,
        background: BackgroundTasks,
        x_hub_signature_256: str | None = Header(default=None),
    ) -> dict[str, str]:
        body = await request.body()
        # La firma se verifica sobre los bytes crudos, antes de interpretar nada.
        if not verify_signature(settings.app_secret, body, x_hub_signature_256):
            raise HTTPException(status_code=401, detail="firma inválida")
        try:
            payload = json.loads(body)
        except ValueError:
            raise HTTPException(status_code=400, detail="JSON inválido") from None
        for msg in parse_messages(payload):
            if not dedup.seen(msg.message_id):
                background.add_task(process, msg)
        return {"status": "ok"}  # Meta espera un 200 rápido; el trabajo sigue en segundo plano

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "cache": cache.stats(), "handoffs_recent": len(handoffs)}

    return app
