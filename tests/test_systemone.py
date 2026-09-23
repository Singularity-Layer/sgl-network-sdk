import json

import httpx
import pytest

from singularity_grid import (
    GridClient,
    SGLAPIError,
    SGLNotFoundError,
    SystemOneChoiceAnswer,
    SystemOneNoulAnswer,
    SystemOneScoreAnswer,
    SystemOneScoreQuestion,
)
from singularity_grid import e2e


def _grid(handler, api_key="scg_test"):
    """GridClient whose HTTP calls go to ``handler`` instead of the network."""
    grid = GridClient(api_key=api_key, base_url="https://grid.test")
    grid._client = httpx.Client(
        base_url="https://grid.test",
        headers=grid._client.headers,
        transport=httpx.MockTransport(handler),
    )
    return grid


RESULT = {
    "object": "systemone.result",
    "model": "convaiinnovations/laya",
    "answers": {
        "route": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.9},
        "urgency": {"type": "score", "score": 0.7},
        "lang": {"type": "noul", "value": "en", "confidence": 0.8},
    },
    "usage": {"input_tokens": 120, "output_tokens": 0, "cost_usd": 0.000001},
}


def test_models_hits_typed_listing_and_filters_client_side():
    seen = {}

    def handler(req):
        seen["url"] = req.url
        seen["key"] = req.headers.get("x-api-key")
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "convaiinnovations/laya", "object": "model", "owned_by": "sgl-network",
             "type": "systemone", "context_window": 16384, "max_questions": 32},
            {"id": "gemma-4-26b", "object": "model", "type": "chat", "context_window": 8192},
        ]})

    models = _grid(handler).systemone.models()
    assert seen["url"].path == "/v1/models"
    assert seen["url"].params["type"] == "systemone"
    assert seen["key"] == "scg_test"
    assert [m.id for m in models] == ["convaiinnovations/laya"]
    assert models[0].max_questions == 32
    assert models[0].context_window == 16384


def test_create_sends_contract_body_and_parses_typed_answers():
    seen = {}

    def handler(req):
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["key"] = req.headers.get("x-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=RESULT)

    res = _grid(handler).systemone.create(
        "laya",
        {"ticket": "I was charged twice"},
        {
            "route": {"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "money", "tech": "bugs"}},
            "urgency": SystemOneScoreQuestion(instructions="How urgent?", criteria=["customer is blocked"]),
            "lang": {"type": "noul", "instructions": "Language code"},
        },
        task="triage",
        tier="standard",
    )

    assert seen["method"] == "POST" and seen["path"] == "/v1/systemone"
    assert seen["key"] == "scg_test"
    assert seen["body"] == {
        "model": "laya",
        "state": {"ticket": "I was charged twice"},
        "questions": {
            "route": {"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "money", "tech": "bugs"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["customer is blocked"]},
            "lang": {"type": "noul", "instructions": "Language code"},
        },
        "task": "triage",
        "tier": "standard",
    }
    # No x402 signing and no session-credit flag: the API key is the credential.
    assert "use_credits" not in seen["body"]

    assert res.object == "systemone.result"
    assert res.model == "convaiinnovations/laya"
    assert isinstance(res.answers["route"], SystemOneChoiceAnswer)
    assert res.answers["route"].choice == "billing"
    assert res.answers["route"].probabilities == {"billing": 0.9, "tech": 0.1}
    assert isinstance(res.answers["urgency"], SystemOneScoreAnswer)
    assert res.answers["urgency"].score == 0.7
    assert isinstance(res.answers["lang"], SystemOneNoulAnswer)
    assert res.answers["lang"].value == "en"
    assert res.usage.input_tokens == 120
    assert res.usage.cost_usd == 0.000001


def test_noul_value_keeps_its_json_type():
    def handler(req):
        body = dict(RESULT, answers={"a": {"type": "noul", "value": True}, "b": {"type": "noul", "value": 3}})
        return httpx.Response(200, json=body)

    res = _grid(handler).systemone.create("laya", {"x": 1}, {"a": {"type": "noul", "instructions": "?"},
                                                             "b": {"type": "noul", "instructions": "?"}})
    assert res.answers["a"].value is True
    assert res.answers["b"].value == 3 and not isinstance(res.answers["b"].value, bool)


def test_402_payment_required_explains_api_key():
    challenge = {
        "x402Version": 1, "accepts": [{}], "price_usd": 0.001,
        "error": {"message": "Payment required. Send X-Payment (x402) or X-API-Key (credits).", "type": "payment_required"},
    }
    grid = _grid(lambda req: httpx.Response(402, json=challenge), api_key=None)
    with pytest.raises(SGLAPIError) as ei:
        grid.systemone.create("laya", {"x": 1}, {"q": {"type": "noul", "instructions": "?"}})
    assert ei.value.status_code == 402
    assert "pass api_key" in str(ei.value)
    assert ei.value.body == challenge


@pytest.mark.parametrize("etype,message", [
    ("insufficient_credits", "Insufficient credits. This request needs ~$0.001000 and your balance is $0.000000."),
    ("pod_cap_reached", "This agent has reached its daily compute cap."),
])
def test_other_402s_keep_server_message(etype, message):
    body = {"error": {"message": message, "type": etype}}
    grid = _grid(lambda req: httpx.Response(402, json=body))
    with pytest.raises(SGLAPIError) as ei:
        grid.systemone.create("laya", {"x": 1}, {"q": {"type": "noul", "instructions": "?"}})
    assert ei.value.status_code == 402
    assert message in str(ei.value)
    assert "pass api_key" not in str(ei.value)
    assert ei.value.body["error"]["type"] == etype


def test_private_create_reserves_then_submits_ciphertext_only():
    node_key = e2e.new_response_keypair()[1]
    calls = []

    def handler(req):
        calls.append((req.url.path, json.loads(req.content)))
        if req.url.path == "/v1/systemone/reserve":
            return httpx.Response(200, json={
                "reservation_token": "sys1.token",
                "node_id": "node-1",
                "node_x25519_pubkey": node_key,
                "node_ed25519_pubkey": None,
                "attestation_verified": True,
                "expires_in_ms": 60000,
            })
        return httpx.Response(402, json={
            "error": {"message": "Insufficient credits.", "type": "insufficient_credits"},
        })

    grid = _grid(handler)
    with pytest.raises(SGLAPIError) as ei:
        grid.systemone.create(
            "laya",
            {"secret": "do not send me"},
            {"q": {"type": "noul", "instructions": "?"}},
            private=True,
        )
    assert ei.value.status_code == 402
    assert "Insufficient credits" in str(ei.value)

    assert calls[0][0] == "/v1/systemone/reserve"
    assert sorted(calls[0][1]) == ["input_tokens_upper_bound", "model"]
    assert calls[0][1]["model"] == "laya"
    assert isinstance(calls[0][1]["input_tokens_upper_bound"], int)

    assert calls[1][0] == "/v1/systemone"
    assert sorted(calls[1][1]) == ["enc", "reservation_token"]
    assert calls[1][1]["reservation_token"] == "sys1.token"
    assert calls[1][1]["enc"]["algorithm"] == e2e.ALGO_V2
    assert isinstance(calls[1][1]["enc"]["ciphertext"], str)
    assert "do not send me" not in json.dumps(calls[1][1])


def test_disabled_endpoint_is_not_found():
    grid = _grid(lambda req: httpx.Response(404, json={"error": "Not found"}))
    with pytest.raises(SGLNotFoundError):
        grid.systemone.create("laya", {"x": 1}, {"q": {"type": "noul", "instructions": "?"}})


def test_existing_surface_unchanged():
    grid = GridClient()
    for name in ("capacity", "models", "pricing", "submit_job", "get_job", "get_attestation",
                 "providers", "chat_completions", "chat_completion_stream", "embeddings", "embed"):
        assert callable(getattr(grid, name))
    assert callable(grid.systemone.models) and callable(grid.systemone.create)
