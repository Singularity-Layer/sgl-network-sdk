"""Anthropic Messages helpers and the pods Sandbox/clone/BYOK surface.

The transport is an httpx.MockTransport, so these pin the wire contract (path, method, headers,
body) and how errors and stream events come back. No network.
"""

import json

import httpx
import pytest

from singularity_grid import ANTHROPIC_VERSION, GridClient, PodsClient
from singularity_grid.errors import SGLAPIError

POD = "11111111-2222-3333-4444-555555555555"


def _grid(handler, api_key="x402c_k"):
    client = GridClient(api_key=api_key)
    headers = dict(client._client.headers)
    client._client.close()
    client._client = httpx.Client(base_url="https://grid.test", headers=headers, transport=httpx.MockTransport(handler))
    return client


def _pods(handler):
    client = PodsClient(api_key="x402c_k", base_url="https://compute.test")
    client._client.close()
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_messages_posts_with_key_and_version():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}})

    out = _grid(handler).messages("qwen-2.5-32b", [{"role": "user", "content": "hello"}], max_tokens=16, system="be brief")
    assert out["content"][0]["text"] == "hi"
    assert seen["url"] == "https://grid.test/v1/messages"
    assert seen["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert seen["headers"]["x-api-key"] == "x402c_k"
    assert seen["body"] == {"model": "qwen-2.5-32b", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, "system": "be brief"}


def test_messages_maps_402_and_reads_anthropic_errors():
    c = _grid(lambda req: httpx.Response(402, json={"type": "error", "error": {"type": "billing_error", "message": "pay"}}), api_key=None)
    with pytest.raises(SGLAPIError) as e:
        c.messages("m", [], max_tokens=1)
    assert e.value.status_code == 402 and "api_key" in str(e.value)
    c = _grid(lambda req: httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "top_p is not supported"}}))
    with pytest.raises(SGLAPIError, match="top_p is not supported"):
        c.messages("m", [], max_tokens=1)


def test_messages_stream_yields_events_until_message_stop():
    sse = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"m"}}\n\n'
        'event: ping\ndata: {"type":"ping"}\n\n'
        'event: content_block_delta\r\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\r\n\r\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n'
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    )
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())

    events = list(_grid(handler).messages_stream("m", [{"role": "user", "content": "x"}], max_tokens=8))
    assert [e["type"] for e in events] == ["message_start", "ping", "content_block_delta", "content_block_delta", "message_stop"]
    assert "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta") == "Hello"
    assert bodies[0]["stream"] is True


def test_messages_stream_raises_on_error_event():
    sse = 'event: error\ndata: {"type":"error","error":{"type":"api_error","message":"node failed"}}\n\n'
    c = _grid(lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode()))
    with pytest.raises(SGLAPIError, match="node failed"):
        list(c.messages_stream("m", [], max_tokens=8))


def test_count_tokens():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        return httpx.Response(200, json={"input_tokens": 4})

    assert _grid(handler).count_tokens("m", [{"role": "user", "content": "hello"}])["input_tokens"] == 4
    assert seen["path"] == "/v1/messages/count_tokens"


def test_create_pod_sends_engine_and_byok_fallbacks():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["idem"] = req.headers.get("idempotency-key")
        return httpx.Response(201, json={"pod": {"id": POD, "status": "provisioning"}})

    ai = {"mode": "byok", "preset": "venice", "api_key": "vk", "managed_models": True,
          "fallbacks": [{"preset": "openrouter", "model": "x/y", "api_key": "ok"}, {"preset": "singularity", "model": "z-ai/glm-5.3"}]}
    _pods(handler).create_pod(engine="pipod", ai=ai)
    assert seen["body"]["engine"] == "pipod"
    assert seen["body"]["ai"]["fallbacks"][1]["preset"] == "singularity"
    assert seen["idem"]


def test_sandbox_routes():
    calls = []

    def handler(req):
        calls.append((req.method, str(req.url), json.loads(req.content) if req.content else None))
        u = str(req.url)
        if u.endswith("/exec"):
            return httpx.Response(200, json={"exec": {"exit_code": 0, "timed_out": False, "stdout": "hi\n", "stderr": "", "truncated": False}})
        if "/files?path=" in u:
            return httpx.Response(200, json={"file": {"path": "a b.txt", "content": "x"}})
        if u.endswith("/files") and req.method == "PUT":
            return httpx.Response(200, json={"file": {"path": "a.txt", "size": 1}})
        if u.endswith("/files"):
            return httpx.Response(200, json={"files": [{"path": "a.txt"}], "root": "/opt/pod/workspace"})
        if u.endswith("/ports") and req.method == "POST":
            return httpx.Response(201, json={"port": {"port": 8080, "visibility": "private", "url": "https://u/", "token": "t"}, "rotated": False})
        if u.endswith("/ports"):
            return httpx.Response(200, json={"ports": [], "limit": 5})
        if u.endswith("/ports/8080"):
            return httpx.Response(200, json={"closed": True, "port": 8080})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "nope"}})

    c = _pods(handler)
    assert c.exec(POD, "echo hi", timeout_seconds=5)["stdout"] == "hi\n"
    assert calls[0] == ("POST", f"https://compute.test/pods/v1/pods/{POD}/exec", {"command": "echo hi", "timeout_seconds": 5})
    assert c.list_files(POD)["root"] == "/opt/pod/workspace"
    assert c.read_file(POD, "a b.txt")["content"] == "x"
    assert calls[2][1].endswith("path=a%20b.txt")
    assert c.write_file(POD, "a.txt", "x")["size"] == 1
    assert calls[3][2] == {"path": "a.txt", "content": "x"}
    assert c.open_port(POD, 8080, "private")["port"]["token"] == "t"
    assert calls[4][2] == {"port": 8080, "visibility": "private"}
    assert c.list_ports(POD)["limit"] == 5
    assert c.close_port(POD, 8080)["closed"] is True
    assert calls[6][0] == "DELETE"


def test_sandbox_error_keeps_code():
    c = _pods(lambda req: httpx.Response(403, json={"error": {"code": "forbidden", "message": "needs pods:control:write"}}))
    with pytest.raises(SGLAPIError, match="pods:control:write") as e:
        c.exec(POD, "ls")
    assert e.value.body["code"] == "forbidden"


def test_clone_pod_and_status():
    seen = {}

    def handler(req):
        if req.method == "POST":
            seen["body"] = json.loads(req.content)
            seen["idem"] = req.headers.get("idempotency-key")
            return httpx.Response(201, json={"pod": {"id": "new"}, "clone": {"source_pod_id": POD, "backup_id": "b", "restore": {"status": "pending"}}})
        return httpx.Response(200, json={"clone": {"status": "restored"}})

    c = _pods(handler)
    out = c.clone_pod(POD, "correct horse", name="copy", idempotency_key="clone-1")
    assert out["clone"]["restore"]["status"] == "pending"
    assert seen == {"body": {"passphrase": "correct horse", "name": "copy"}, "idem": "clone-1"}
    assert c.clone_status("new")["status"] == "restored"
