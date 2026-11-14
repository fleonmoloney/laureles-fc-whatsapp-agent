"""Integración con la WhatsApp Cloud API: firma de webhooks, lectura de mensajes y envío.

AVISO: el formato de los eventos y del endpoint de envío sigue la documentación de Meta tal como la
conocía el autor al escribir este módulo. NO se probó contra un número real ni se contrastó con la
documentación vigente. Antes de producción, verificarlo con el entorno de pruebas de Meta.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

import httpx

MAX_TEXT = 4096  # largo máximo de un mensaje de texto


def sign(app_secret: str, body: bytes) -> str:
    """Valor del encabezado X-Hub-Signature-256 para `body` (útil para pruebas y simulaciones)."""
    return "sha256=" + hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()


def verify_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    """Verifica en tiempo constante la firma HMAC-SHA256 sobre los bytes crudos del cuerpo."""
    if not header or not header.startswith("sha256="):
        return False
    return hmac.compare_digest(header.encode(), sign(app_secret, body).encode())


@dataclass(frozen=True)
class IncomingMessage:
    wa_id: str
    message_id: str
    text: str | None  # None si el mensaje no es de texto (audio, imagen, etc.)


def parse_messages(payload: Any) -> list[IncomingMessage]:
    """Extrae los mensajes entrantes. Ignora estados de entrega y cualquier estructura inesperada."""
    out: list[IncomingMessage] = []
    if not isinstance(payload, dict):
        return out
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes") or []:
            value = change.get("value") if isinstance(change, dict) else None
            if not isinstance(value, dict):
                continue
            for msg in value.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                wa_id, message_id = msg.get("from"), msg.get("id")
                if not isinstance(wa_id, str) or not isinstance(message_id, str):
                    continue
                text = None
                if msg.get("type") == "text":
                    body = (msg.get("text") or {}).get("body")
                    if isinstance(body, str) and body.strip():
                        text = body.strip()
                out.append(IncomingMessage(wa_id, message_id, text))
    return out


class WhatsAppClient:
    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        api_version: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._http = httpx.Client(
            base_url=f"https://graph.facebook.com/{api_version}",
            headers={"Authorization": f"Bearer {access_token}"},
            transport=transport,
            timeout=timeout,
        )
        self._path = f"/{phone_number_id}/messages"

    def send_text(self, to: str, body: str) -> None:
        resp = self._http.post(
            self._path,
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": body[:MAX_TEXT]},
            },
        )
        resp.raise_for_status()

    def close(self) -> None:
        self._http.close()
