from pathlib import Path

import pytest

from src.config import Settings
from src.main import create_app

ROOT = Path(__file__).resolve().parent.parent
SECRET = "app-secret"
OPERATOR = "59899000000"
ACK = "Tu consulta pasó a secretaría; te responden en horario de atención."


class FakeWA:
    """Reemplaza al cliente de WhatsApp: registra lo que se enviaría."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.closed = False
        self.fail = False

    def send_text(self, to: str, body: str) -> None:
        if self.fail:
            raise RuntimeError("WhatsApp caído")
        self.sent.append((to, body))

    def close(self) -> None:
        self.closed = True


def make_settings(**over) -> Settings:
    base = dict(
        verify_token="verify-me",
        app_secret=SECRET,
        access_token="token-de-prueba",
        phone_number_id="123456",
        api_version="vTEST",
        faq_path=str(ROOT / "data" / "faq.example.json"),
        operator_wa_id=OPERATOR,
        handoff_ack=ACK,
        rate_limit_per_minute=0,  # sin límite salvo en la prueba que lo ejerce
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def wa() -> FakeWA:
    return FakeWA()


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def app(settings, wa):
    return create_app(settings, wa_client=wa)
