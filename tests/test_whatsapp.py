import json

import httpx
import pytest

from src.whatsapp import MAX_TEXT, IncomingMessage, WhatsAppClient, parse_messages, sign, verify_signature


def test_signature_roundtrip_and_failures():
    body = b'{"a": 1}'
    good = sign("secreto", body)
    assert good.startswith("sha256=")
    assert verify_signature("secreto", body, good)
    assert not verify_signature("otro", body, good)
    assert not verify_signature("secreto", body + b" ", good)
    assert not verify_signature("secreto", body, None)
    assert not verify_signature("secreto", body, "")
    assert not verify_signature("secreto", body, good.replace("sha256=", "sha1="))
    assert not verify_signature("secreto", body, "sha256=ñandú")  # no ASCII: no debe explotar


def event(*messages):
    return {"entry": [{"changes": [{"value": {"messages": list(messages)}}]}]}


def test_parse_text_message():
    msgs = parse_messages(event({"from": "598", "id": "w1", "type": "text", "text": {"body": "  hola  "}}))
    assert msgs == [IncomingMessage("598", "w1", "hola")]


def test_parse_non_text_message_keeps_the_sender_but_no_text():
    assert parse_messages(event({"from": "598", "id": "w2", "type": "image"})) == [IncomingMessage("598", "w2", None)]


def test_parse_ignores_status_updates_and_blank_text():
    status = {"entry": [{"changes": [{"value": {"statuses": [{"id": "w3", "status": "read"}]}}]}]}
    assert parse_messages(status) == []
    blank = parse_messages(event({"from": "598", "id": "w4", "type": "text", "text": {"body": "   "}}))
    assert blank[0].text is None


@pytest.mark.parametrize("payload", [None, [], "x", {}, {"entry": "x"}, {"entry": [None, 3]},
                                     {"entry": [{"changes": [None, {"value": "x"}, {"value": {"messages": [1, {}]}}]}]}])
def test_parse_survives_malformed_payloads(payload):
    assert parse_messages(payload) == []


def test_parse_multiple_messages_in_one_event():
    m = [{"from": "1", "id": "a", "type": "text", "text": {"body": "uno"}},
         {"from": "2", "id": "b", "type": "text", "text": {"body": "dos"}}]
    assert [x.text for x in parse_messages(event(*m))] == ["uno", "dos"]


def test_client_sends_the_documented_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    c = WhatsAppClient("tok-123", "555", "vTEST", transport=httpx.MockTransport(handler))
    c.send_text("598111", "hola")
    assert seen["url"] == "https://graph.facebook.com/vTEST/555/messages"
    assert seen["auth"] == "Bearer tok-123"
    assert seen["body"] == {"messaging_product": "whatsapp", "to": "598111", "type": "text", "text": {"body": "hola"}}


def test_client_truncates_long_text_and_raises_on_http_errors():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["text"]["body"])
        return httpx.Response(200 if len(sent) == 1 else 401, json={})

    c = WhatsAppClient("t", "1", "v", transport=httpx.MockTransport(handler))
    c.send_text("1", "x" * (MAX_TEXT + 500))
    assert len(sent[0]) == MAX_TEXT
    with pytest.raises(httpx.HTTPStatusError):
        c.send_text("1", "otra vez")
