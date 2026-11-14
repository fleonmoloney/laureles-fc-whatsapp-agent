import json

import pytest
from fastapi.testclient import TestClient

from src.main import NON_TEXT_REPLY, create_app
from src.whatsapp import sign
from tests.conftest import ACK, OPERATOR, SECRET, make_settings

SOCIO = "59891111111"
FAQ_Q = "¿Cuánto sale la cuota social?"  # texto canónico del FAQ: puntaje 1.0
FAQ_A = "La cuota social mensual es de $1.200 para socios mayores y $600 para menores de 14 años."


def text_event(text, *, wa_id=SOCIO, mid="wamid.1", type_="text"):
    msg = {"from": wa_id, "id": mid, "type": type_}
    if type_ == "text":
        msg["text"] = {"body": text}
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "WABA", "changes": [{"field": "messages", "value": {
            "messaging_product": "whatsapp", "metadata": {"phone_number_id": "123456"}, "messages": [msg]}}]}],
    }


def post(client, payload, *, secret=SECRET, signature=None, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    sig = signature if signature is not None else sign(secret, body)
    if sig:
        headers["X-Hub-Signature-256"] = sig
    return client.post("/webhook", content=body, headers=headers)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# --- verificación del webhook (GET) --------------------------------------------------------------

def test_verification_handshake_echoes_the_challenge(client):
    r = client.get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "1158201444"})
    assert r.status_code == 200
    assert r.text == "1158201444"


@pytest.mark.parametrize("params", [
    {"hub.mode": "subscribe", "hub.verify_token": "otro", "hub.challenge": "1"},
    {"hub.mode": "unsubscribe", "hub.verify_token": "verify-me", "hub.challenge": "1"},
    {"hub.mode": "subscribe", "hub.verify_token": "verify-me"},
    {},
])
def test_verification_is_rejected_when_anything_is_off(client, params):
    assert client.get("/webhook", params=params).status_code == 403


# --- firma ----------------------------------------------------------------------------------------

def test_invalid_signature_is_rejected_and_nothing_is_sent(client, wa):
    r = post(client, text_event(FAQ_Q), secret="no-es-el-secreto")
    assert r.status_code == 401
    assert wa.sent == []


def test_missing_signature_is_rejected(client, wa):
    assert post(client, text_event(FAQ_Q), signature="").status_code == 401
    assert wa.sent == []


def test_signature_does_not_cover_a_tampered_body(client, wa):
    original = json.dumps(text_event(FAQ_Q)).encode()
    tampered = json.dumps(text_event("otra cosa")).encode()
    assert post(client, None, raw=tampered, signature=sign(SECRET, original)).status_code == 401
    assert wa.sent == []


def test_signed_garbage_is_a_400_not_a_crash(client):
    assert post(client, None, raw=b"esto no es json").status_code == 400


# --- flujo principal --------------------------------------------------------------------------------

def test_faq_question_is_answered_to_the_sender(client, wa):
    assert post(client, text_event(FAQ_Q)).status_code == 200
    assert wa.sent == [(SOCIO, FAQ_A)]


def test_out_of_scope_question_is_handed_to_a_human_not_answered(client, wa):
    post(client, text_event("¿Quién es el técnico del primer equipo?"))
    sent_to = [to for to, _ in wa.sent]
    assert sent_to == [SOCIO, OPERATOR]
    assert wa.sent[0][1] == ACK  # el socio recibe el aviso, no una respuesta inventada
    assert "técnico" in wa.sent[1][1] and SOCIO in wa.sent[1][1]
    assert client.get("/health").json()["handoffs_recent"] == 1


def test_handoff_without_ack_sends_nothing_to_the_member(wa):
    app = create_app(make_settings(handoff_ack=None, operator_wa_id=None), wa_client=wa)
    with TestClient(app) as c:
        post(c, text_event("¿Quién es el técnico del primer equipo?"))
    assert wa.sent == []  # queda solo en el registro de derivaciones


def test_status_notifications_are_acknowledged_and_ignored(client, wa):
    status = {"entry": [{"changes": [{"field": "messages", "value": {"statuses": [{"id": "wamid.9", "status": "delivered"}]}}]}]}
    assert post(client, status).status_code == 200
    assert wa.sent == []


def test_non_text_messages_get_a_clear_reply(client, wa):
    post(client, text_event(None, type_="audio"))
    assert wa.sent == [(SOCIO, NON_TEXT_REPLY)]


def test_meta_retries_do_not_duplicate_the_answer(client, wa):
    event = text_event(FAQ_Q, mid="wamid.same")
    post(client, event)
    post(client, event)  # reintento con el mismo id
    assert len(wa.sent) == 1


def test_repeated_question_is_served_from_the_cache(client, wa):
    post(client, text_event(FAQ_Q, mid="wamid.a", wa_id="59891111111"))
    post(client, text_event(FAQ_Q, mid="wamid.b", wa_id="59892222222"))
    assert [b for _, b in wa.sent] == [FAQ_A, FAQ_A]
    assert client.get("/health").json()["cache"]["hits"] == 1


def test_sender_rate_limit_caps_the_work_done(wa):
    app = create_app(make_settings(rate_limit_per_minute=2), wa_client=wa)
    with TestClient(app) as c:
        for i in range(4):
            post(c, text_event(FAQ_Q, mid=f"wamid.{i}"))
    assert len(wa.sent) == 2


def test_a_failing_send_does_not_break_the_service(client, wa):
    wa.fail = True
    assert post(client, text_event(FAQ_Q, mid="wamid.x")).status_code == 200
    wa.fail = False
    post(client, text_event(FAQ_Q, mid="wamid.y"))
    assert len(wa.sent) == 1


def test_health_and_client_is_closed_on_shutdown(app, wa):
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
    assert wa.closed
