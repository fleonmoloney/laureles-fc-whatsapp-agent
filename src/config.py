"""Configuración leída del entorno. Falla al arrancar si falta algo obligatorio."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_REQUIRED = {
    "WHATSAPP_VERIFY_TOKEN": "token que vos elegís y cargás en el panel de Meta",
    "WHATSAPP_APP_SECRET": "App Secret de la app de Meta (firma de los webhooks)",
    "WHATSAPP_ACCESS_TOKEN": "token de acceso de WhatsApp Cloud API",
    "WHATSAPP_PHONE_NUMBER_ID": "ID del número emisor",
    "WHATSAPP_API_VERSION": "versión vigente de la Graph API (ver documentación de Meta)",
    "FAQ_PATH": "archivo JSON con las preguntas frecuentes reales del club",
}


@dataclass(frozen=True)
class Settings:
    verify_token: str
    app_secret: str
    access_token: str
    phone_number_id: str
    api_version: str
    faq_path: str
    operator_wa_id: str | None = None
    handoff_ack: str | None = None
    confidence_threshold: float = 0.8
    cache_threshold: float = 0.95
    cache_ttl_seconds: int = 6 * 3600
    cache_max_entries: int = 1000
    rate_limit_per_minute: int = 10  # mensajes por remitente; 0 desactiva el límite
    embedder: str = "tfidf"  # "tfidf" (sin descargas) | "minilm" (sentence-transformers)
    embedder_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str | None = None  # si está y hay ANTHROPIC_API_KEY, redacta con LLM

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        missing = [f"  {k}: {why}" for k, why in _REQUIRED.items() if not env.get(k)]
        if missing:
            raise RuntimeError("Faltan variables de entorno:\n" + "\n".join(missing))
        return cls(
            verify_token=env["WHATSAPP_VERIFY_TOKEN"],
            app_secret=env["WHATSAPP_APP_SECRET"],
            access_token=env["WHATSAPP_ACCESS_TOKEN"],
            phone_number_id=env["WHATSAPP_PHONE_NUMBER_ID"],
            api_version=env["WHATSAPP_API_VERSION"],
            faq_path=env["FAQ_PATH"],
            operator_wa_id=env.get("OPERATOR_WA_ID") or None,
            handoff_ack=env.get("HANDOFF_ACK") or None,
            confidence_threshold=float(env.get("CONFIDENCE_THRESHOLD", 0.8)),
            cache_threshold=float(env.get("CACHE_THRESHOLD", 0.95)),
            cache_ttl_seconds=int(env.get("CACHE_TTL_SECONDS", 6 * 3600)),
            cache_max_entries=int(env.get("CACHE_MAX_ENTRIES", 1000)),
            rate_limit_per_minute=int(env.get("RATE_LIMIT_PER_MINUTE", 10)),
            embedder=env.get("EMBEDDER", "tfidf"),
            embedder_model=env.get("EMBEDDER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            llm_model=env.get("LLM_MODEL") or None,
        )
