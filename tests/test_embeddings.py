import base64
import hashlib
import json

import httpx
import pytest

from singularity_grid import (
    EMBEDDINGGEMMA2_LIMITS,
    EMBEDDINGGEMMA2_MIME_TYPES,
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
from singularity_grid.embeddings import validate_embedding_request


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
    ("extension", "mime_type"),
    [
        ("jpg", "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("png", "image/png"),
        ("webp", "image/webp"),
        ("wav", "audio/wav"),
        ("flac", "audio/flac"),
        ("mp3", "audio/mpeg"),
        ("mp4", "video/mp4"),
    ],
)
def test_file_helper_normalizes_every_supported_extension(tmp_path, extension, mime_type):
    path = tmp_path / f"fixture.{extension}"
    path.write_bytes(b"fixture")
    assert media_from_file(path)["mime_type"] == mime_type


def test_explicit_platform_audio_mime_aliases_are_normalized():
    assert media_from_bytes(b"wav", mime_type="audio/x-wav")["mime_type"] == "audio/wav"
    assert media_from_bytes(b"flac", mime_type="audio/x-flac")["mime_type"] == "audio/flac"


def test_public_mime_and_cardinality_limits_match_the_grid_contract():
    assert EMBEDDINGGEMMA2_MIME_TYPES == {
        "image": ("image/jpeg", "image/png", "image/webp"),
        "audio": ("audio/wav", "audio/flac", "audio/mpeg"),
        "video": ("video/mp4",),
    }
    assert EMBEDDINGGEMMA2_LIMITS["max_audio_parts_per_item"] == 1
    assert EMBEDDINGGEMMA2_LIMITS["max_video_parts_per_item"] == 1


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
            "encoding_format": "float",
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
            encoding_format="float",
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


def test_legacy_broad_input_is_not_coerced_by_the_new_typed_surface():
    legacy_input = {"ids": [1, 2, 3]}

    def handler(request):
        payload = json.loads(request.content)
        assert payload["input"] == legacy_input
        assert payload["input_type"] == "provider-specific"
        return httpx.Response(200, json={"data": [], "usage": {}})

    with _mock_client(handler) as client:
        response = client.embeddings(
            "custom-legacy-model",
            legacy_input,
            input_type="provider-specific",
        )
    assert response["data"] == []


def test_non_float_encoding_is_rejected_before_transport():
    requested = False

    def handler(_request):
        nonlocal requested
        requested = True
        return httpx.Response(200, json={"data": [], "usage": {}})

    with _mock_client(handler) as client:
        with pytest.raises(EmbeddingInputError) as raised:
            client.embeddings(
                "nomic-embed-text-v1.5",
                "text",
                encoding_format="base64",
            )
    assert raised.value.code == "invalid_encoding_format"
    assert requested is False


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


def test_duplicate_audio_and_video_parts_have_distinct_stable_codes():
    audio = audio_part(
        media_from_bytes(b"wav", mime_type="audio/wav"),
        duration_seconds=1,
    )
    video = video_part(
        media_from_bytes(b"mp4", mime_type="video/mp4"),
        duration_seconds=1,
    )
    with pytest.raises(EmbeddingInputError) as duplicate_audio:
        multimodal_item(audio, audio)
    assert duplicate_audio.value.code == "too_many_audio_parts"
    with pytest.raises(EmbeddingInputError) as duplicate_video:
        multimodal_item(video, video)
    assert duplicate_video.value.code == "too_many_video_parts"


@pytest.mark.parametrize(
    ("input_type", "prefix"),
    [
        (None, "task: search result | query: "),
        ("query", "task: search result | query: "),
        ("document", "title: none | text: "),
        ("unspecified", ""),
    ],
)
def test_context_preflight_matches_grid_prefix_and_template_budget(input_type, prefix):
    text_bytes = (
        EMBEDDINGGEMMA2_LIMITS["max_processed_tokens_per_item"]
        - len(prefix.encode("utf-8"))
        - EMBEDDINGGEMMA2_LIMITS["processor_template_tokens"]
    )
    validate_embedding_request(
        EMBEDDINGGEMMA2_MODEL,
        "x" * text_bytes,
        dimensions=768,
        input_type=input_type,
        encoding_format="float",
    )
    with pytest.raises(EmbeddingInputError) as raised:
        validate_embedding_request(
            EMBEDDINGGEMMA2_MODEL,
            "x" * (text_bytes + 1),
            dimensions=768,
            input_type=input_type,
            encoding_format="float",
        )
    assert raised.value.code == "context_limit_exceeded"


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
