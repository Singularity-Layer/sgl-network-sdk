"""run_with_payment relays x402 v1 and v2 headers both ways."""

import httpx
import pytest

from singularity_grid import ProcessorsClient, PROCESSORS_BASE_URL
from singularity_grid.errors import SGLAPIError

V1 = "eyJ4NDAyVmVyc2lvbiI6MX0="  # {"x402Version":1}
V2 = "eyJ4NDAyVmVyc2lvbiI6Mn0="  # {"x402Version":2}


def _client(handler):
    c = ProcessorsClient(api_key="x402c_test")
    c._client = httpx.Client(
        base_url=PROCESSORS_BASE_URL,
        headers={"Accept": "application/json", "X-API-Key": "x402c_test"},
        transport=httpx.MockTransport(handler),
    )
    return c


def test_v1_request_headers_unchanged():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"ok": True}, headers={"X-PAYMENT-RESPONSE": "v1res"})

    c = _client(handler)
    assert c.run_with_payment("p", {"a": 1}, payment_header=V1) == {"ok": True}
    assert seen["x-payment"] == V1
    assert "payment-signature" not in seen
    assert "x-api-key" not in seen  # buyer lane never sends the management key
    assert c.last_payment_response == "v1res"


def test_v2_sends_payment_signature_and_reads_payment_response():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"run": "r1"}, headers={"PAYMENT-RESPONSE": "v2res", "X-PAYMENT-RESPONSE": "v1res"})

    c = _client(handler)
    assert c.run_with_payment("p", {}, payment_signature=V2) == {"run": "r1"}
    assert seen["payment-signature"] == V2
    assert "x-payment" not in seen
    assert c.last_payment_response == "v2res"


def test_402_exposes_payment_required_offers():
    def handler(request):
        return httpx.Response(402, json={"error": "payment_required", "accepts": []}, headers={"PAYMENT-REQUIRED": "offers"})

    c = _client(handler)
    with pytest.raises(SGLAPIError) as exc:
        c.run_with_payment("p", {})
    assert exc.value.status_code == 402
    assert exc.value.payment_required == "offers"
    assert exc.value.body["accepts"] == []
    assert c.last_payment_response is None


def test_v1_402_without_header_has_no_offers():
    c = _client(lambda r: httpx.Response(402, json={"accepts": []}))
    with pytest.raises(SGLAPIError) as exc:
        c.run_with_payment("p", {})
    assert exc.value.payment_required is None
    assert exc.value.headers == {}


def test_last_payment_response_resets_between_calls():
    responses = [
        httpx.Response(200, json={}, headers={"PAYMENT-RESPONSE": "first"}),
        httpx.Response(200, json={"x": 1}),
    ]
    c = _client(lambda r: responses.pop(0))
    c.run_with_payment("p", {}, payment_signature=V2)
    assert c.last_payment_response == "first"
    c.run_with_payment("p", {}, payment_signature=V2)
    assert c.last_payment_response is None


def test_api_error_positional_signature_still_works():
    e = SGLAPIError(500, "boom", {"code": "x"})
    assert e.headers == {} and e.payment_required is None and e.code == "x"
