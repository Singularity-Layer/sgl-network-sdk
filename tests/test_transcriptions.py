import base64
import hashlib
import json
import struct
import uuid

import base58
import httpx
import pytest
from nacl import bindings
from nacl.signing import SigningKey

from singularity_grid import (
    GridClient,
    SGLConnectionError,
    SGLAPIError,
    TRANSCRIPTION_MAX_PCM_BYTES,
    TRANSCRIPTION_MODEL_REVISION,
    TRANSCRIPTION_MODEL_SHA256,
    TranscriptionInputError,
    TranscriptionResponseError,
    validate_transcription_pcm,
)
from singularity_grid import e2e

NODE_ID = "11111111-2222-4333-8444-555566667777"
REQUEST_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
JOB_ID = REQUEST_ID
NODE_SECRET = bytes([7]) * 32
NODE_PUBLIC = base58.b58encode(bindings.crypto_scalarmult_base(NODE_SECRET)).decode()
ED_SIGNING = SigningKey(bytes([9]) * 32)
ED_PUBLIC = base58.b58encode(bytes(ED_SIGNING.verify_key)).decode()


def _keybind_signature(version=1, transport_key=NODE_PUBLIC):
    message = (
        b"SGL-NODE-KEYBIND-v1"
        + b"\x00"
        + uuid.UUID(NODE_ID).bytes
        + bytes(ED_SIGNING.verify_key)
        + base58.b58decode(transport_key)
        + struct.pack("<I", version)
    )
    return base58.b58encode(ED_SIGNING.sign(message).signature).decode()


def _reservation(request, mutate=None):
    value = {
        "reservation_token": "signed-reservation",
        "request_id": request["request_id"],
        "model": request["model"],
        "model_revision": TRANSCRIPTION_MODEL_REVISION,
        "model_sha256": TRANSCRIPTION_MODEL_SHA256,
        "transcription_protocol": request["transcription_protocol"],
        "sample_count": request["sample_count"],
        "language": request["language"],
        "node_id": NODE_ID,
        "node_x25519_pubkey": NODE_PUBLIC,
        "node_ed25519_pubkey": ED_PUBLIC,
        "node_x25519_pubkey_sig": _keybind_signature(),
        "key_version": 1,
        "tee_type": "tdx",
        "attestation_verified": True,
        "expires_in_ms": 60_000,
        "quote": {
            "audio_seconds": request["sample_count"] / 16_000,
            "sample_count": request["sample_count"],
            "rate_usd_per_second": 0.0001,
            "minimum_charge_usd": 0.0001,
            "price_usd": max(100, (request["sample_count"] + 159) // 160) / 1e6,
            "currency": "USDC",
        },
    }
    if mutate:
        mutate(value)
    return value


def _decrypt_request(envelope):
    shared = bindings.crypto_scalarmult(
        NODE_SECRET, base58.b58decode(envelope["client_ephemeral_pubkey"])
    )
    key = e2e._hkdf(shared, e2e._INFO_INPUT)
    aad = e2e._aad_input(
        NODE_PUBLIC,
        envelope["client_ephemeral_pubkey"],
        envelope["client_response_pubkey"],
    )
    blob = base64.b64decode(envelope["ciphertext"], validate=True)
    plaintext = bindings.crypto_aead_xchacha20poly1305_ietf_decrypt(
        blob[24:], aad, blob[:24], key
    )
    return json.loads(plaintext)


def _seal_result(response_public, result, plaintext_bytes=None):
    ephemeral_secret = bytes([11]) * 32
    ephemeral_public = base58.b58encode(
        bindings.crypto_scalarmult_base(ephemeral_secret)
    ).decode()
    shared = bindings.crypto_scalarmult(
        ephemeral_secret, base58.b58decode(response_public)
    )
    key = e2e._hkdf(shared, e2e._INFO_OUTPUT)
    aad = e2e._aad_output(response_public, ephemeral_public)
    nonce = bytes([13]) * 24
    plaintext = plaintext_bytes if plaintext_bytes is not None else json.dumps(result, separators=(",", ":")).encode()
    encrypted = bindings.crypto_aead_xchacha20poly1305_ietf_encrypt(
        plaintext, aad, nonce, key
    )
    return ephemeral_public, base64.b64encode(nonce + encrypted).decode()


def _result_signature(ciphertext):
    digest = hashlib.sha256(ciphertext.encode()).hexdigest()
    message = f"sgl-result-v1\n{JOB_ID}\ntranscription\n{digest}".encode()
    return base58.b58encode(ED_SIGNING.sign(message).signature).decode()


def _run_exchange(
    *,
    pcm=None,
    mutate_reservation=None,
    mutate_result=None,
    mutate_envelope=None,
    throw_on_submit=False,
    submit_counter=None,
    submit_status=None,
    plaintext_bytes=None,
    response_body=...,
    transport_error=None,
):
    raw = pcm if pcm is not None else bytes(index % 251 for index in range(320))
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request, body))
        if len(calls) == 1:
            return httpx.Response(
                200, json=_reservation(body, mutate_reservation)
            )
        if submit_counter is not None:
            submit_counter.append(1)
        if response_body is not ...:
            return httpx.Response(200, content=json.dumps(response_body), headers={"content-type": "application/json"})
        if transport_error is not None:
            raise transport_error
        if throw_on_submit:
            raise httpx.ConnectError("connection closed after submit")
        if submit_status is not None:
            return httpx.Response(submit_status, json={"error": {"code": "outcome_unknown", "type": "payment_error"}})
        inner = _decrypt_request(body["enc"])
        result = {
            "object": "transcription",
            "protocol": "transcription-v1",
            "request_id": inner["request_id"],
            "job_id": JOB_ID,
            "model": inner["model"],
            "model_revision": inner["model_revision"],
            "model_sha256": inner["model_sha256"],
            "sample_count": inner["audio"]["sample_count"],
            "text": "Synthetic transcript.",
            "language_hint": inner["language"],
            "language": "en",
            "duration_seconds": inner["audio"]["sample_count"] / 16_000,
            "segments": [
                {
                    "start": 0,
                    "end": inner["audio"]["sample_count"] / 16_000,
                    "text": "Synthetic transcript.",
                }
            ],
        }
        if mutate_result:
            mutate_result(result)
        ephemeral_public, ciphertext = _seal_result(
            body["enc"]["client_response_pubkey"], result, plaintext_bytes
        )
        response = {
            "object": "transcription",
            "job_id": JOB_ID,
            "sealed_result": {
                "algorithm": e2e.ALGO_V2,
                "encoding": "base64",
                "ephemeral_public_key": ephemeral_public,
                "ciphertext": ciphertext,
            },
            "result_envelope_signature": _result_signature(ciphertext),
            "result_envelope_version": "v1",
            "usage": {"audio_seconds": inner["audio"]["sample_count"] / 16_000, "cost_usd": max(100, (inner["audio"]["sample_count"] + 159) // 160) / 1e6},
        }
        if mutate_envelope:
            mutate_envelope(response)
        return httpx.Response(200, json=response)

    client = GridClient(
        api_key="x402c_test",
        base_url="https://grid.test",
        transcription_canary_token="private-test-token",
    )
    client._client.close()
    client._client = httpx.Client(
        base_url="https://grid.test", transport=httpx.MockTransport(handler)
    )
    try:
        response = client.transcribe_pcm(raw, request_id=REQUEST_ID, language="en")
        return response, calls
    finally:
        client.close()


def test_transcribe_pcm_seals_audio_and_returns_verified_bound_result():
    pcm = bytes(index % 251 for index in range(320))
    response, calls = _run_exchange(pcm=pcm)
    assert len(calls) == 2
    assert calls[0][0].url.path == "/v1/audio/transcriptions/reserve"
    assert calls[0][1] == {
        "model": "whisper-1",
        "model_revision": TRANSCRIPTION_MODEL_REVISION,
        "model_sha256": TRANSCRIPTION_MODEL_SHA256,
        "transcription_protocol": "transcription-v1",
        "request_id": REQUEST_ID,
        "sample_rate": 16_000,
        "channels": 1,
        "bits_per_sample": 16,
        "sample_count": 160,
        "language": "en",
        "use_credits": True,
    }
    assert calls[1][0].url.path == "/v1/audio/transcriptions"
    assert list(calls[1][1]) == ["reservation_token", "enc"]
    assert calls[1][1]["enc"]["algorithm"] == e2e.ALGO_V2
    assert calls[1][1]["enc"]["encoding"] == "base64"
    assert base64.b64encode(pcm) not in calls[1][0].content
    inner = _decrypt_request(calls[1][1]["enc"])
    assert list(inner) == ["protocol", "request_id", "model", "model_revision", "model_sha256", "language", "audio"]
    assert list(inner["audio"]) == ["format", "sample_rate", "channels", "bits_per_sample", "sample_count", "data"]
    assert inner["model_revision"] == TRANSCRIPTION_MODEL_REVISION
    assert inner["model_sha256"] == TRANSCRIPTION_MODEL_SHA256
    assert inner["audio"]["sample_count"] == 160
    assert inner["audio"]["data"] == base64.b64encode(pcm).decode()
    assert response["text"] == "Synthetic transcript."
    assert response["attestation"]["verified"] is True
    assert calls[0][0].headers["x-sgl-stt-canary"] == "private-test-token"
    assert calls[1][0].headers["x-sgl-stt-canary"] == "private-test-token"


def test_pcm_bounds_and_language_fail_before_network():
    with pytest.raises(TranscriptionInputError) as empty:
        validate_transcription_pcm(b"")
    assert empty.value.code == "empty_audio"
    with pytest.raises(TranscriptionInputError) as odd:
        validate_transcription_pcm(b"123")
    assert odd.value.code == "invalid_pcm_length"
    with pytest.raises(TranscriptionInputError) as container:
        validate_transcription_pcm(b"RIFF0000WAVE")
    assert container.value.code == "unsupported_audio_container"
    with pytest.raises(TranscriptionInputError) as large:
        validate_transcription_pcm(b"0" * (TRANSCRIPTION_MAX_PCM_BYTES + 2))
    assert large.value.code == "audio_too_long"

    requested = False

    def handler(_request):
        nonlocal requested
        requested = True
        return httpx.Response(500)

    client = GridClient(base_url="https://grid.test")
    client._client.close()
    client._client = httpx.Client(
        base_url="https://grid.test", transport=httpx.MockTransport(handler)
    )
    with client:
        with pytest.raises(TranscriptionInputError) as language:
            client.transcribe_pcm(b"00", language="ENG")
    assert language.value.code == "invalid_language"
    assert requested is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(request_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"),
        lambda value: value.update(model="other"),
        lambda value: value.update(model_revision="b" * 40),
        lambda value: value.update(model_sha256="b" * 64),
        lambda value: value.update(sample_count=value["sample_count"] + 1),
        lambda value: value.update(language="fr"),
        lambda value: value.update(
            node_x25519_pubkey_sig=base58.b58encode(bytes(64)).decode()
        ),
        lambda value: value.update(key_version=2),
    ],
)
def test_reservation_substitution_is_rejected_before_submit(mutate):
    submit_calls = []
    with pytest.raises(TranscriptionResponseError):
        _run_exchange(mutate_reservation=mutate, submit_counter=submit_calls)
    assert submit_calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(request_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"),
        lambda value: value.update(model="other"),
        lambda value: value.update(model_revision="b" * 40),
        lambda value: value.update(model_sha256="b" * 64),
        lambda value: value.update(sample_count=value["sample_count"] + 1),
        lambda value: value.update(language_hint="fr"),
    ],
)
def test_sealed_result_binding_substitution_is_rejected(mutate):
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(mutate_result=mutate)
    assert raised.value.code == "binding_mismatch"


def test_bad_signature_and_unsupported_algorithm_or_encoding_are_refused():
    with pytest.raises(TranscriptionResponseError) as signature:
        _run_exchange(
            mutate_envelope=lambda value: value.update(
                result_envelope_signature=base58.b58encode(bytes(64)).decode()
            )
        )
    assert signature.value.code == "unverified_result"

    for field in ("algorithm", "encoding"):
        with pytest.raises(TranscriptionResponseError) as envelope:
            _run_exchange(
                mutate_envelope=lambda value, field=field: value[
                    "sealed_result"
                ].update({field: "unsupported"})
            )
        assert envelope.value.code == "invalid_envelope"


def test_submit_transport_failure_is_not_retried():
    submit_calls = []
    with pytest.raises(SGLConnectionError):
        _run_exchange(throw_on_submit=True, submit_counter=submit_calls)
    assert len(submit_calls) == 1


def test_pcm_file_is_bounded_before_network(tmp_path):
    path = tmp_path / "too-large.pcm"
    path.write_bytes(b"0" * (TRANSCRIPTION_MAX_PCM_BYTES + 1))
    with GridClient(base_url="https://grid.test") as client:
        with pytest.raises(TranscriptionInputError) as raised:
            client.transcribe_pcm_file(path)
    assert raised.value.code == "file_too_large"


@pytest.mark.parametrize("mutate", [
    lambda r: r["quote"].update(audio_seconds=1),
    lambda r: r["quote"].update(rate_usd_per_second=0.01),
    lambda r: r["quote"].update(minimum_charge_usd=0),
    lambda r: r["quote"].update(price_usd=0.0002),
    lambda r: r.update(node_id=r["node_id"].replace("-", "")),
    lambda r: r.update(sample_count=True),
])
def test_invalid_quote_or_node_never_submits(mutate):
    submits = []
    with pytest.raises(TranscriptionResponseError):
        _run_exchange(mutate_reservation=mutate, submit_counter=submits)
    assert submits == []


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(text="a" * 65537),
    lambda r: r.update(text="unsafe\u0000"),
    lambda r: r.update(text="unsafe\u0085"),
    lambda r: r.update(text="\ud800"),
    lambda r: r["segments"][0].update(text="\udc00"),
    lambda r: r.update(language="ENG"),
    lambda r: r.pop("language"),
    lambda r: r.update(duration_seconds=2),
    lambda r: r["segments"][0].update(end=0.2),
    lambda r: r.update(segments=[r["segments"][0]] * 257),
    lambda r: r.update(job_id=NODE_ID),
    lambda r: r.update(sample_count=True),
])
def test_signed_invalid_result_is_rejected(mutate):
    with pytest.raises(TranscriptionResponseError):
        _run_exchange(mutate_result=mutate)


@pytest.mark.parametrize("status", [402, 409, 502, 504])
def test_submit_payment_errors_preserve_identity_and_are_never_retried(status):
    submits = []
    with pytest.raises(SGLAPIError) as raised:
        _run_exchange(submit_status=status, submit_counter=submits)
    assert raised.value.status_code == status
    assert raised.value.code == "outcome_unknown"
    assert submits == [1]


def test_invalid_utf8_is_rejected_after_authenticated_decryption():
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(plaintext_bytes=b"\xff")
    assert raised.value.code == "invalid_envelope"


def test_signature_is_checked_before_decrypting_invalid_output_key():
    def mutate(r):
        r["sealed_result"]["ephemeral_public_key"] = "1" * 32
        r["result_envelope_signature"] = base58.b58encode(bytes(64)).decode()
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(mutate_envelope=mutate)
    assert raised.value.code == "unverified_result"


def test_signed_noncanonical_base64_and_wrong_envelope_shape_are_rejected():
    def noncanonical(r):
        r["sealed_result"]["ciphertext"] += "\n"
        r["result_envelope_signature"] = _result_signature(r["sealed_result"]["ciphertext"])
    for mutate in [noncanonical, lambda r: r.update(object="other"),
                   lambda r: r.update(result_envelope_version="v2"),
                   lambda r: r["sealed_result"].update(ciphertext="A" * 1400000)]:
        with pytest.raises(TranscriptionResponseError) as raised:
            _run_exchange(mutate_envelope=mutate)
        assert raised.value.code == "invalid_envelope"


@pytest.mark.parametrize("samples", [1, 16000, 16001, 16080, 32000, 960000])
def test_fractional_billing_and_micro_usdc_rounding(samples):
    response, _ = _run_exchange(pcm=bytes(samples * 2))
    assert response["usage"]["audio_seconds"] == samples / 16000
    assert response["usage"]["cost_usd"] == max(100, (samples + 159) // 160) / 1e6


@pytest.mark.parametrize("mutate", [
    lambda r: r["usage"].update(audio_seconds=1),
    lambda r: r["usage"].update(audio_seconds=True),
    lambda r: r["usage"].update(cost_usd=0),
    lambda r: r["usage"].update(cost_usd=0.0002),
    lambda r: r.update(billing_pending="true"),
])
def test_billing_metadata_must_match_exact_audio_and_price(mutate):
    with pytest.raises(TranscriptionResponseError):
        _run_exchange(mutate_envelope=mutate)


def test_negative_pcm_samples_are_not_mistaken_for_mp3_frame_sync():
    assert validate_transcription_pcm(b"\xff\xff\xff\xff") == 2


def test_stt_uses_120_second_timeout_without_changing_other_routes():
    with GridClient(timeout=1.234) as client:
        assert client._transcription_timeout == 120
        assert client._client.timeout.read == 1.234
    with GridClient(transcription_timeout=180) as client:
        assert client._transcription_timeout == 180


def test_transcription_http_timeout_is_attached_to_reserve_and_submit():
    _, calls = _run_exchange()
    for request, _ in calls:
        assert request.extensions["timeout"]["read"] == 120


@pytest.mark.parametrize("body", [None, [], True, "invalid"])
def test_non_object_response_bodies_produce_typed_response_failure(body):
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(response_body=body)
    assert raised.value.code == "invalid_envelope"


@pytest.mark.parametrize("error", [httpx.ReadError("closed"), httpx.ReadTimeout("late")])
def test_ambiguous_read_failure_exposes_reconciliation_id_and_never_retries(error):
    submits = []
    with pytest.raises(SGLConnectionError) as raised:
        _run_exchange(transport_error=error, submit_counter=submits)
    assert REQUEST_ID in str(raised.value)
    assert "reconcile" in str(raised.value)
    assert submits == [1]


def test_signed_low_order_transport_key_is_rejected_before_submit():
    submits = []
    def mutate(r):
        r["node_x25519_pubkey"] = base58.b58encode(bytes(32)).decode()
        r["node_x25519_pubkey_sig"] = _keybind_signature(1, r["node_x25519_pubkey"])
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(mutate_reservation=mutate, submit_counter=submits)
    assert raised.value.code == "invalid_reservation"
    assert submits == []


def test_oversized_base58_keys_are_rejected_before_expensive_decode():
    with pytest.raises(TranscriptionResponseError) as reservation:
        _run_exchange(mutate_reservation=lambda r: r.update(node_x25519_pubkey="1" * 10000))
    assert reservation.value.code == "invalid_reservation"
    with pytest.raises(TranscriptionResponseError) as envelope:
        _run_exchange(mutate_envelope=lambda r: r["sealed_result"].update(ephemeral_public_key="1" * 10000))
    assert envelope.value.code == "invalid_envelope"


@pytest.mark.parametrize("request_id", [REQUEST_ID.upper(), REQUEST_ID.replace("4ccc", "1ccc"), REQUEST_ID.replace("4ccc", "7ccc"), REQUEST_ID.replace("-", "")])
def test_request_id_must_be_canonical_lowercase_uuid4_before_network(request_id):
    with GridClient(base_url="https://grid.test") as client:
        requested = []
        client._request = lambda *args, **kwargs: requested.append(True)
        with pytest.raises(TranscriptionInputError) as raised:
            client.transcribe_pcm(b"\x00\x00", request_id=request_id)
        assert raised.value.code == "invalid_request_id"
        assert requested == []


def test_signed_silence_with_empty_text_and_segments_is_valid():
    response, _ = _run_exchange(mutate_result=lambda r: r.update(text="", segments=[], language=None))
    assert response["text"] == ""
    assert response["segments"] == []
    assert response["usage"]["audio_seconds"] == 0.01


@pytest.mark.parametrize("text", ["a" * 32769, "é" * 16385])
def test_aggregate_segment_utf8_bytes_are_bounded(text):
    segments = [{"start": 0, "end": 0.004, "text": text}, {"start": 0.004, "end": 0.01, "text": text}]
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(mutate_result=lambda r: r.update(segments=segments))
    assert raised.value.code == "invalid_result"


def test_aggregate_segment_utf8_bytes_accept_exact_limit():
    text = "é" * 16384
    segments = [{"start": 0, "end": 0.004, "text": text}, {"start": 0.004, "end": 0.01, "text": text}]
    response, _ = _run_exchange(mutate_result=lambda r: r.update(segments=segments))
    assert len(response["segments"]) == 2


@pytest.mark.parametrize("segments", [
    [{"start": 0.005, "end": 0.01, "text": "a"}, {"start": 0.004, "end": 0.009, "text": "b"}],
    [{"start": 0, "end": 0.01, "text": "a"}, {"start": 0.001, "end": 0.009, "text": "b"}],
    [{"start": 0.005, "end": 0.008, "text": "a"}, {"start": 0.004, "end": 0.01, "text": "b"}],
])
def test_segment_starts_and_ends_cannot_go_backward_within_overlap_tolerance(segments):
    with pytest.raises(TranscriptionResponseError) as raised:
        _run_exchange(mutate_result=lambda r: r.update(segments=segments))
    assert raised.value.code == "invalid_result"


def test_forward_moving_segments_can_have_small_overlap():
    segments = [{"start": 0, "end": 0.006, "text": "a"}, {"start": 0.004, "end": 0.01, "text": "b"}]
    response, _ = _run_exchange(mutate_result=lambda r: r.update(segments=segments))
    assert len(response["segments"]) == 2


def test_valid_zero_duration_segments_can_keep_equal_timestamps():
    segments = [{"start": 0, "end": 0, "text": "a"}, {"start": 0, "end": 0, "text": "b"}]
    response, _ = _run_exchange(mutate_result=lambda r: r.update(segments=segments))
    assert len(response["segments"]) == 2
