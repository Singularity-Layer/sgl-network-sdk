"""Bounded client-sealed helpers for the private transcription-v1 contract."""

from __future__ import annotations

import base64
import json
import math
import re
import uuid
from typing import Any, Dict, Literal, Mapping, Optional

from .errors import TranscriptionInputError, TranscriptionResponseError
from .models import TranscriptionResponse, TranscriptionSegment, TranscriptionUsage

TRANSCRIPTION_PROTOCOL: Literal["transcription-v1"] = "transcription-v1"
TRANSCRIPTION_MODEL: Literal["whisper-1"] = "whisper-1"
TRANSCRIPTION_MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
TRANSCRIPTION_MODEL_SHA256 = (
    "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"
)
TRANSCRIPTION_AUDIO_FORMAT = "pcm_s16le"
TRANSCRIPTION_SAMPLE_RATE = 16_000
TRANSCRIPTION_CHANNELS = 1
TRANSCRIPTION_BITS_PER_SAMPLE = 16
TRANSCRIPTION_MAX_DURATION_SECONDS = 60
TRANSCRIPTION_MAX_SAMPLES = (
    TRANSCRIPTION_SAMPLE_RATE * TRANSCRIPTION_MAX_DURATION_SECONDS
)
TRANSCRIPTION_MAX_PCM_BYTES = TRANSCRIPTION_MAX_SAMPLES * 2
TRANSCRIPTION_MAX_TEXT_BYTES = 64 * 1024
TRANSCRIPTION_MAX_SEGMENTS = 256
TRANSCRIPTION_MAX_RESULT_ENVELOPE_BYTES = 1024 * 1024
TRANSCRIPTION_RATE_USD_PER_SECOND = 0.0001
TRANSCRIPTION_MINIMUM_CHARGE_USD = 0.0001


def transcription_price_usd(sample_count: int) -> float:
    # Compute micro-USDC from integer samples before converting to dollars.
    return max(100, (sample_count + 159) // 160) / 1_000_000

_UUID_V4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_LANGUAGE = re.compile(r"^(?:auto|[a-z]{2})$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _valid_text(value: Any) -> bool:
    if not isinstance(value, str) or _CONTROL_CHARACTERS.search(value):
        return False
    try:
        return len(value.encode("utf-8", errors="strict")) <= TRANSCRIPTION_MAX_TEXT_BYTES
    except UnicodeEncodeError:
        return False


def normalize_transcription_pcm(pcm: Any) -> bytes:
    """Copy a bytes-like PCM buffer and enforce the v1 limits locally."""
    if not isinstance(pcm, (bytes, bytearray, memoryview)):
        raise TranscriptionInputError("invalid_audio_type", "Audio must be raw PCM bytes")
    size = pcm.nbytes if isinstance(pcm, memoryview) else len(pcm)
    if size > TRANSCRIPTION_MAX_PCM_BYTES:
        raise TranscriptionInputError("audio_too_long", "Audio exceeds the PCM byte limit")
    raw = bytes(pcm)
    if not raw:
        raise TranscriptionInputError(
            "empty_audio", "Audio must contain at least one PCM sample"
        )
    if len(raw) % 2:
        raise TranscriptionInputError(
            "invalid_pcm_length",
            "Signed 16-bit PCM must contain a whole number of two-byte samples",
        )
    is_riff_wave = len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    is_mp4 = len(raw) >= 8 and raw[4:8] == b"ftyp"
    is_mp3 = raw[:3] == b"ID3"
    if is_riff_wave or raw[:4] in (b"fLaC", b"OggS") or is_mp4 or is_mp3:
        raise TranscriptionInputError(
            "unsupported_audio_container",
            "Audio containers are not accepted; decode to raw mono 16 kHz "
            "signed 16-bit PCM first",
        )
    if len(raw) > TRANSCRIPTION_MAX_PCM_BYTES:
        raise TranscriptionInputError(
            "audio_too_long",
            f"Audio must be at most {TRANSCRIPTION_MAX_DURATION_SECONDS} seconds "
            f"({TRANSCRIPTION_MAX_PCM_BYTES} bytes)",
        )
    return raw


def validate_transcription_pcm(pcm: Any) -> int:
    """Return the authoritative sample count for valid v1 raw PCM."""
    return len(normalize_transcription_pcm(pcm)) // 2


def validate_transcription_options(
    *,
    model: str,
    language: str,
    request_id: Optional[str],
    use_credits: bool,
    node: Optional[str],
    max_price: Optional[float],
) -> Dict[str, Any]:
    if model != TRANSCRIPTION_MODEL:
        raise TranscriptionInputError(
            "invalid_model", f"model must be {TRANSCRIPTION_MODEL}"
        )
    if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
        raise TranscriptionInputError(
            "invalid_language",
            'language must be "auto" or a lowercase two-letter code',
        )
    logical_id = request_id if request_id is not None else str(uuid.uuid4())
    if not isinstance(logical_id, str) or not _UUID_V4.fullmatch(logical_id):
        raise TranscriptionInputError(
            "invalid_request_id", "request_id must be a canonical lowercase UUIDv4"
        )
    if not isinstance(use_credits, bool):
        raise TranscriptionInputError(
            "invalid_option", "use_credits must be a boolean"
        )
    if node is not None and (
        not isinstance(node, str) or not node or len(node) > 128
    ):
        raise TranscriptionInputError(
            "invalid_option", "node must be a nonempty bounded identifier"
        )
    if max_price is not None and (
        isinstance(max_price, bool)
        or not isinstance(max_price, (int, float))
        or not math.isfinite(max_price)
        or max_price <= 0
    ):
        raise TranscriptionInputError(
            "invalid_option", "max_price must be a positive finite number"
        )
    return {
        "model": model,
        "language": language,
        "request_id": logical_id,
        "use_credits": use_credits,
        "node": node,
        "max_price": max_price,
    }


def validate_transcription_reservation(
    value: Any,
    *,
    request_id: str,
    model: str,
    language: str,
    sample_count: int,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription reservation is not an object"
        )
    required_strings = (
        "reservation_token",
        "request_id",
        "model",
        "model_revision",
        "model_sha256",
        "transcription_protocol",
        "language",
        "node_id",
        "node_x25519_pubkey",
        "node_ed25519_pubkey",
        "node_x25519_pubkey_sig",
    )
    if any(not isinstance(value.get(field), str) or not value[field] for field in required_strings):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription reservation is incomplete"
        )
    if (len(value["reservation_token"]) > 16_384
        or any(not 32 <= len(value[field]) <= 44 for field in ("node_x25519_pubkey", "node_ed25519_pubkey"))
        or not 64 <= len(value["node_x25519_pubkey_sig"]) <= 88):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription reservation keys or token exceed their bounds"
        )
    if (
        value["request_id"] != request_id
        or value["model"] != model
        or value["language"] != language
        or type(value.get("sample_count")) is not int
        or value.get("sample_count") != sample_count
        or value["transcription_protocol"] != TRANSCRIPTION_PROTOCOL
    ):
        raise TranscriptionResponseError(
            "binding_mismatch", "Transcription reservation does not match the request"
        )
    if (
        not _REVISION.fullmatch(value["model_revision"])
        or value["model_revision"] != TRANSCRIPTION_MODEL_REVISION
    ):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription model revision is invalid"
        )
    if (
        not _SHA256.fullmatch(value["model_sha256"])
        or value["model_sha256"] != TRANSCRIPTION_MODEL_SHA256
    ):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription model hash is invalid"
        )
    key_version = value.get("key_version")
    expires = value.get("expires_in_ms")
    if (
        isinstance(key_version, bool)
        or not isinstance(key_version, int)
        or key_version < 0
        or key_version > 0xFFFFFFFF
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or expires <= 0
    ):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription key version or expiry is invalid"
        )
    quote = value.get("quote")
    expected_audio_seconds = sample_count / TRANSCRIPTION_SAMPLE_RATE
    if (
        not isinstance(quote, dict)
        or isinstance(quote.get("audio_seconds"), bool)
        or quote.get("audio_seconds") != expected_audio_seconds
        or type(quote.get("sample_count")) is not int
        or quote.get("sample_count") != sample_count
        or quote.get("rate_usd_per_second") != TRANSCRIPTION_RATE_USD_PER_SECOND
        or quote.get("minimum_charge_usd") != TRANSCRIPTION_MINIMUM_CHARGE_USD
        or quote.get("currency") != "USDC"
        or isinstance(quote.get("price_usd"), bool)
        or not isinstance(quote.get("price_usd"), (int, float))
        or not math.isfinite(quote["price_usd"])
        or abs(quote["price_usd"] - transcription_price_usd(sample_count)) > 1e-9
    ):
        raise TranscriptionResponseError(
            "binding_mismatch", "Transcription quote does not match the reserved audio"
        )
    if not isinstance(value.get("attestation_verified"), bool) or (
        value.get("tee_type") is not None
        and not isinstance(value.get("tee_type"), str)
    ):
        raise TranscriptionResponseError(
            "invalid_reservation", "Transcription attestation metadata is invalid"
        )
    return value


def create_transcription_plaintext(
    pcm: bytes, reservation: Mapping[str, Any]
) -> bytes:
    """Create the canonical, AEAD-authenticated inner request."""
    payload = {
        "protocol": TRANSCRIPTION_PROTOCOL,
        "request_id": reservation["request_id"],
        "model": reservation["model"],
        "model_revision": reservation["model_revision"],
        "model_sha256": reservation["model_sha256"],
        "language": reservation["language"],
        "audio": {
            "format": TRANSCRIPTION_AUDIO_FORMAT,
            "sample_rate": TRANSCRIPTION_SAMPLE_RATE,
            "channels": TRANSCRIPTION_CHANNELS,
            "bits_per_sample": TRANSCRIPTION_BITS_PER_SAMPLE,
            "sample_count": reservation["sample_count"],
            "data": base64.b64encode(pcm).decode("ascii"),
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_usage(
    value: Any, sample_count: int, quote: Any
) -> TranscriptionUsage:
    if not isinstance(value, dict):
        raise TranscriptionResponseError(
            "invalid_result", "Transcription usage is invalid"
        )
    audio_seconds = value.get("audio_seconds")
    cost_usd = value.get("cost_usd")
    billable_seconds = sample_count / TRANSCRIPTION_SAMPLE_RATE
    if (
        isinstance(audio_seconds, bool)
        or audio_seconds != billable_seconds
        or isinstance(cost_usd, bool)
        or not isinstance(cost_usd, (int, float))
        or not math.isfinite(cost_usd)
        or cost_usd < 0
    ):
        raise TranscriptionResponseError(
            "binding_mismatch",
            "Transcription billing metadata does not match the audio",
        )
    if isinstance(quote, dict) and (
        abs(cost_usd - transcription_price_usd(sample_count)) > 1e-9
        or abs(cost_usd - quote.get("price_usd", -1)) > 1e-9
    ):
        raise TranscriptionResponseError(
            "binding_mismatch", "Transcription billing does not match the reserved quote"
        )
    return {"audio_seconds": audio_seconds, "cost_usd": cost_usd}


def validate_transcription_result(
    value: Any,
    *,
    outer_job_id: str,
    reservation: Mapping[str, Any],
    usage_value: Any,
    billing_pending_value: Any,
) -> TranscriptionResponse:
    if not isinstance(value, dict):
        raise TranscriptionResponseError(
            "invalid_result", "Decrypted transcription result is not an object"
        )
    if not (
        value.get("object") == "transcription"
        and value.get("protocol") == TRANSCRIPTION_PROTOCOL
        and value.get("request_id") == reservation["request_id"]
        and value.get("job_id") == outer_job_id
        and value.get("model") == reservation["model"]
        and value.get("model_revision") == reservation["model_revision"]
        and value.get("model_sha256") == reservation["model_sha256"]
        and type(value.get("sample_count")) is int
        and value.get("sample_count") == reservation["sample_count"]
        and value.get("language_hint") == reservation["language"]
    ):
        raise TranscriptionResponseError(
            "binding_mismatch", "Transcription result does not match its reservation"
        )
    text = value.get("text")
    if not isinstance(text, str) or not _valid_text(text):
        raise TranscriptionResponseError(
            "invalid_result", "Transcription text is invalid"
        )
    language = value.get("language")
    if "language" not in value or (language is not None and (
        not isinstance(language, str) or not re.fullmatch(r"[a-z]{2}", language)
    )):
        raise TranscriptionResponseError(
            "invalid_result", "Detected language is invalid"
        )
    duration = value.get("duration_seconds")
    input_seconds = reservation["sample_count"] / TRANSCRIPTION_SAMPLE_RATE
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
        or duration > min(input_seconds + 1, TRANSCRIPTION_MAX_DURATION_SECONDS + 1)
    ):
        raise TranscriptionResponseError(
            "binding_mismatch", "Transcription duration contradicts the audio"
        )
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) > TRANSCRIPTION_MAX_SEGMENTS:
        raise TranscriptionResponseError(
            "invalid_result", "Transcription segments are invalid"
        )
    segments: list[TranscriptionSegment] = []
    last_start = 0.0
    last_end = 0.0
    segment_text_bytes = 0
    for segment in raw_segments:
        if not isinstance(segment, dict):
            raise TranscriptionResponseError(
                "invalid_result", "Transcription segments are invalid"
            )
        start, end, segment_text = (
            segment.get("start"),
            segment.get("end"),
            segment.get("text"),
        )
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not isinstance(segment_text, str)
            or not _valid_text(segment_text)
            or start < 0
            or end < start
            or start < last_start
            or end < last_end
            or start < last_end - 0.05
            or end > input_seconds + 1
            or end > duration + 0.05
        ):
            raise TranscriptionResponseError(
                "invalid_result", "Transcription segments are invalid"
            )
        segment_text_bytes += len(segment_text.encode("utf-8"))
        if segment_text_bytes > TRANSCRIPTION_MAX_TEXT_BYTES:
            raise TranscriptionResponseError(
                "invalid_result", "Aggregate transcription segment text exceeds its byte limit"
            )
        last_start = float(start)
        last_end = float(end)
        segments.append({"start": float(start), "end": float(end), "text": segment_text})
    if billing_pending_value is not None and not isinstance(billing_pending_value, bool):
        raise TranscriptionResponseError(
            "invalid_result", "Transcription billing state is invalid"
        )
    result: TranscriptionResponse = {
        "object": "transcription",
        "job_id": outer_job_id,
        "request_id": reservation["request_id"],
        "model": TRANSCRIPTION_MODEL,
        "model_revision": reservation["model_revision"],
        "model_sha256": reservation["model_sha256"],
        "transcription_protocol": TRANSCRIPTION_PROTOCOL,
        "sample_count": reservation["sample_count"],
        "text": text,
        "language_hint": reservation["language"],
        "language": language,
        "duration_seconds": float(duration),
        "segments": segments,
        "usage": _validate_usage(usage_value, reservation["sample_count"], reservation["quote"]),
        "attestation": {
            "node_id": reservation["node_id"],
            "tee_type": reservation.get("tee_type"),
            "verified": reservation.get("attestation_verified") is True,
        },
    }
    if billing_pending_value is True:
        result["billing_pending"] = True
    return result
