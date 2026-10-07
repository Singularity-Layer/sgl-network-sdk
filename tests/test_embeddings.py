import base64
import hashlib
import json

import httpx
import pytest

from singularity_grid import (
    EMBEDDINGGEMMA2_LIMITS,
    EMBEDDINGGEMMA2_MODEL,
    EmbeddingInputError,
    GridClient,
    SGLAPIError,
    audio_part,
    image_part,
    media_from_base64,
    media_from_bytes,
    media_from_file,
    multimodal_item,
    text_part,
    video_part,
)


def _mock_client(handler):
    client = GridClient(api_key="scg_test", base_url="https://grid.test")
    client._client.close()
    client._client = httpx.Client(
        base_url="https://grid.test",
        transport=httpx.MockTransport(handler),
    )
    return client


def test_media_helpers_create_canonical_base64_and_sha(tmp_path):
    raw = b"small-media-fixture"
    expected_sha = hashlib.sha256(raw).hexdigest()

    from_bytes = media_from_bytes(raw, mime_type="image/png")
    assert from_bytes == {
        "encoding": "base64",
        "mime_type": "image/png",
        "data": base64.b64encode(raw).decode("ascii"),
        "sha256": expected_sha,
    }
    assert media_from_base64(
        from_bytes["data"], mime_type="image/png", sha256=expected_sha
    ) == from_bytes

    path = tmp_path / "fixture.png"
    path.write_bytes(raw)
    assert media_from_file(path) == from_bytes


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (lambda: media_from_base64("a", mime_type="image/png"), "invalid_base64"),
        (
            lambda: media_from_base64("YQ==", mime_type="image/png", sha256="0" * 64),
            "sha256_mismatch",
        ),
        (lambda: media_from_bytes(b"x", mime_type="image/svg+xml"), "unsupported_mime_type"),
        (lambda: media_from_file("/definitely/missing.png"), "file_not_found"),
    ],
)
def test_media_helpers_return_stable_local_error_codes(call, code):
    with pytest.raises(EmbeddingInputError) as raised:
        call()
    assert raised.value.code == code


def test_multimodal_request_preserves_batch_and_part_order():
    image = media_from_bytes(b"png", mime_type="image/png")
    audio = media_from_bytes(b"wav", mime_type="audio/wav")
    video = media_from_bytes(b"mp4", mime_type="video/mp4")
    item = multimodal_item(
        text_part("before"),
        image_part(image),
        audio_part(audio, duration_seconds=1.25),
        video_part(video, duration_seconds=2),
        text_part("after"),
    )

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert payload == {
            "model": EMBEDDINGGEMMA2_MODEL,
            "input": [item, "legacy text in the same batch"],
            "dimensions": 256,
            "input_type": "unspecified",
            "tier": "standard",
        }
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                ],
                "model": EMBEDDINGGEMMA2_MODEL,
                "usage": {
                    "prompt_tokens": 445,
                    "total_tokens": 445,
                    "cost_usd": 0.000001,
                    "breakdown": {"text": 25, "image": 280, "audio": 0, "video": 140},
                },
                "processor_revision": "a" * 40,
                "embedding_protocol": "embedding-multimodal-v1",
            },
        )

    with _mock_client(handler) as client:
        response = client.embeddings(
            EMBEDDINGGEMMA2_MODEL,
            [item, "legacy text in the same batch"],
            dimensions=256,
            input_type="unspecified",
            tier="standard",
        )
    assert response["usage"]["breakdown"]["image"] == 280
    assert response["processor_revision"] == "a" * 40


@pytest.mark.parametrize("input_value", ["one", ["one", "two"]])
def test_existing_string_inputs_keep_their_wire_shape(input_value):
    def handler(request):
        assert json.loads(request.content)["input"] == input_value
        return httpx.Response(200, json={"data": [], "usage": {}})

    with _mock_client(handler) as client:
        response = client.embeddings("nomic-embed-text-v1.5", input_value, dimensions=64)
    assert response["data"] == []


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"dimensions": 64}, "invalid_dimensions"),
        ({"input_type": "bad"}, "invalid_input_type"),
    ],
)
def test_embeddinggemma_request_validation_has_stable_codes(kwargs, code):
    with GridClient(base_url="https://grid.test") as client:
        with pytest.raises(EmbeddingInputError) as raised:
            client.embeddings(EMBEDDINGGEMMA2_MODEL, "text", **kwargs)
    assert raised.value.code == code


def test_batch_and_duration_limits_fail_before_network():
    audio = media_from_bytes(b"wav", mime_type="audio/wav")
    with pytest.raises(EmbeddingInputError) as duration:
        audio_part(audio, duration_seconds=31)
    assert duration.value.code == "invalid_duration"

    with GridClient(base_url="https://grid.test") as client:
        with pytest.raises(EmbeddingInputError) as batch:
            client.embeddings(EMBEDDINGGEMMA2_MODEL, ["x"] * 17)
    assert batch.value.code == "invalid_batch"


def test_encoded_body_limit_is_checked_before_network():
    # A valid 16 MiB MP4 fits the media cap but its base64 envelope exceeds the 24 MiB body
    # cap once combined with a second file. Use repeated bytes so this stays deterministic.
    video = media_from_bytes(
        b"v" * EMBEDDINGGEMMA2_LIMITS["max_video_bytes"],
        mime_type="video/mp4",
    )
    image = media_from_bytes(b"i" * (3 * 1024 * 1024), mime_type="image/png")
    item = multimodal_item(
        video_part(video, duration_seconds=1),
        image_part(image),
    )
    with GridClient(base_url="https://grid.test") as client:
        with pytest.raises(EmbeddingInputError) as raised:
            client.embeddings(EMBEDDINGGEMMA2_MODEL, [item])
    assert raised.value.code == "body_too_large"


def test_api_error_exposes_server_stable_code():
    def handler(_request):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Embedding input could not be processed.",
                    "type": "invalid_request_error",
                    "code": "embedding_input_invalid",
                }
            },
        )

    with _mock_client(handler) as client:
        with pytest.raises(SGLAPIError) as raised:
            client.embeddings(EMBEDDINGGEMMA2_MODEL, "text")
    assert raised.value.code == "embedding_input_invalid"
    assert raised.value.status_code == 400
