"""Typed EmbeddingGemma 2 inputs and bounded local media helpers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import mimetypes
import os
import stat
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from .errors import EmbeddingInputError
from .models import (
    EmbeddingAudioPart,
    EmbeddingContentPart,
    EmbeddingImagePart,
    EmbeddingInput,
    EmbeddingTextPart,
    EmbeddingVideoPart,
    InlineEmbeddingMedia,
    MultimodalEmbeddingItem,
)

EMBEDDINGGEMMA2_MODEL = "embeddinggemma-2"
EMBEDDINGGEMMA2_DIMENSIONS = (768, 512, 256, 128)
EMBEDDINGGEMMA2_PROTOCOL = "embedding-multimodal-v1"

EMBEDDINGGEMMA2_LIMITS = {
    "max_body_bytes": 24 * 1024 * 1024,
    "max_batch_items": 16,
    "max_parts_per_item": 16,
    "max_images_per_item": 8,
    "max_image_bytes_per_item": 8 * 1024 * 1024,
    "max_image_pixels": 16_000_000,
    "max_audio_bytes": 8 * 1024 * 1024,
    "max_audio_seconds": 30,
    "max_video_bytes": 16 * 1024 * 1024,
    "max_video_seconds": 32,
    "max_video_frames": 32,
    "max_request_media_bytes": 20 * 1024 * 1024,
    "max_processed_tokens_per_item": 8192,
    "max_total_text_bytes": 10 * 1024 * 1024,
    "processor_template_tokens": 12,
    "image_tokens": 280,
    "video_frame_tokens": 140,
    "video_frames_per_second": 1,
}

_MIME_TO_MODALITY = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "audio/wav": "audio",
    "audio/flac": "audio",
    "audio/mpeg": "audio",
    "video/mp4": "video",
}
_MIME_LIMITS = {
    "image": EMBEDDINGGEMMA2_LIMITS["max_image_bytes_per_item"],
    "audio": EMBEDDINGGEMMA2_LIMITS["max_audio_bytes"],
    "video": EMBEDDINGGEMMA2_LIMITS["max_video_bytes"],
}


def _media_limit(mime_type: str) -> int:
    modality = _MIME_TO_MODALITY.get(mime_type)
    if modality is None:
        raise EmbeddingInputError(
            "unsupported_mime_type",
            f"Unsupported EmbeddingGemma 2 MIME type: {mime_type!r}",
        )
    return _MIME_LIMITS[modality]


def media_from_bytes(data: bytes, *, mime_type: str) -> InlineEmbeddingMedia:
    """Create an inline media envelope, enforcing the model's per-file limit."""

    if not isinstance(data, bytes):
        raise EmbeddingInputError("invalid_media", "Media data must be bytes")
    limit = _media_limit(mime_type)
    if not data:
        raise EmbeddingInputError("invalid_media", "Media data must not be empty")
    if len(data) > limit:
        raise EmbeddingInputError(
            "media_too_large",
            f"{mime_type} media is {len(data)} bytes; the limit is {limit} bytes",
        )
    return {
        "encoding": "base64",
        "mime_type": mime_type,
        "data": base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def media_from_base64(
    data: str,
    *,
    mime_type: str,
    sha256: Optional[str] = None,
) -> InlineEmbeddingMedia:
    """Validate canonical base64, enforce size, and calculate or verify SHA-256."""

    limit = _media_limit(mime_type)
    if not isinstance(data, str) or not data:
        raise EmbeddingInputError("invalid_base64", "Base64 media must be a nonempty string")
    if len(data) > 4 * ((limit + 2) // 3):
        raise EmbeddingInputError("media_too_large", f"{mime_type} media exceeds the {limit}-byte limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EmbeddingInputError("invalid_base64", "Media is not canonical base64") from exc
    if base64.b64encode(raw).decode("ascii") != data:
        raise EmbeddingInputError("invalid_base64", "Media is not canonical base64")
    if not raw:
        raise EmbeddingInputError("invalid_media", "Media data must not be empty")
    if len(raw) > limit:
        raise EmbeddingInputError(
            "media_too_large",
            f"{mime_type} media is {len(raw)} bytes; the limit is {limit} bytes",
        )
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and sha256 != digest:
        raise EmbeddingInputError("sha256_mismatch", "Media SHA-256 does not match its bytes")
    return {"encoding": "base64", "mime_type": mime_type, "data": data, "sha256": digest}


def media_from_file(
    path: Union[str, os.PathLike[str]],
    *,
    mime_type: Optional[str] = None,
) -> InlineEmbeddingMedia:
    """Read one bounded regular local file and return its base64 + SHA-256 envelope.

    The helper never fetches URLs and reads at most the selected modality limit plus one byte.
    Pass ``mime_type`` when a file extension is missing or ambiguous.
    """

    file_path = Path(path)
    selected_mime = mime_type or mimetypes.guess_type(file_path.name)[0]
    if selected_mime is None:
        raise EmbeddingInputError(
            "unsupported_mime_type",
            "Could not infer a supported MIME type; pass mime_type explicitly",
        )
    limit = _media_limit(selected_mime)
    try:
        with file_path.open("rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise EmbeddingInputError("file_not_regular", f"Media path is not a regular file: {file_path}")
            if metadata.st_size > limit:
                raise EmbeddingInputError(
                    "media_too_large",
                    f"{selected_mime} media is {metadata.st_size} bytes; the limit is {limit} bytes",
                )
            raw = handle.read(limit + 1)
    except FileNotFoundError as exc:
        raise EmbeddingInputError("file_not_found", f"Media file does not exist: {file_path}") from exc
    except OSError as exc:
        raise EmbeddingInputError("file_read_failed", f"Could not read media file: {file_path}") from exc
    return media_from_bytes(raw, mime_type=selected_mime)


def text_part(text: str) -> EmbeddingTextPart:
    if not isinstance(text, str) or not text:
        raise EmbeddingInputError("invalid_text", "Text parts must be nonempty strings")
    return {"type": "text", "text": text}


def image_part(media: InlineEmbeddingMedia) -> EmbeddingImagePart:
    _validate_media(media, "image")
    return {"type": "image", "media": media}


def audio_part(media: InlineEmbeddingMedia, *, duration_seconds: float) -> EmbeddingAudioPart:
    _validate_duration(duration_seconds, "audio")
    _validate_media(media, "audio")
    return {"type": "audio", "media": media, "duration_seconds": duration_seconds}


def video_part(media: InlineEmbeddingMedia, *, duration_seconds: float) -> EmbeddingVideoPart:
    _validate_duration(duration_seconds, "video")
    _validate_media(media, "video")
    return {"type": "video", "media": media, "duration_seconds": duration_seconds}


def multimodal_item(*parts: EmbeddingContentPart) -> MultimodalEmbeddingItem:
    """Create one vector item while preserving the supplied content-part order."""

    item: MultimodalEmbeddingItem = {"content": list(parts)}
    _validate_item(item)
    return item


def _validate_duration(value: Any, modality: str) -> None:
    maximum = (
        EMBEDDINGGEMMA2_LIMITS["max_audio_seconds"]
        if modality == "audio"
        else EMBEDDINGGEMMA2_LIMITS["max_video_seconds"]
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or value > maximum:
        raise EmbeddingInputError(
            "invalid_duration",
            f"{modality} duration_seconds must be greater than 0 and at most {maximum}",
        )


def _validate_media(media: Any, expected_modality: str) -> int:
    if not isinstance(media, dict) or set(media) != {"encoding", "mime_type", "data", "sha256"}:
        raise EmbeddingInputError("invalid_media", "Media must contain encoding, mime_type, data, and sha256")
    if media.get("encoding") != "base64":
        raise EmbeddingInputError("invalid_media", "Only inline base64 media is supported")
    mime_type = media.get("mime_type")
    if not isinstance(mime_type, str) or _MIME_TO_MODALITY.get(mime_type) != expected_modality:
        raise EmbeddingInputError("unsupported_mime_type", f"Media MIME type does not match {expected_modality}")
    digest = media.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise EmbeddingInputError("invalid_sha256", "Media sha256 must be 64 lowercase hexadecimal characters")
    encoded = media.get("data")
    if not isinstance(encoded, str):
        raise EmbeddingInputError("invalid_base64", "Base64 media must be a nonempty string")
    normalized = media_from_base64(encoded, mime_type=mime_type, sha256=digest)
    return len(base64.b64decode(normalized["data"], validate=True))


def _validate_item(item: Any) -> Tuple[int, int, int, bool]:
    if not isinstance(item, dict) or set(item) != {"content"}:
        raise EmbeddingInputError("invalid_item", "Each multimodal item must contain only content")
    parts = item.get("content")
    max_parts = EMBEDDINGGEMMA2_LIMITS["max_parts_per_item"]
    if not isinstance(parts, list) or not parts or len(parts) > max_parts:
        raise EmbeddingInputError("invalid_parts", f"Each item requires 1-{max_parts} ordered content parts")
    images = image_bytes = audio = video = total_bytes = text_bytes = media_tokens = 0
    for part in parts:
        if not isinstance(part, dict):
            raise EmbeddingInputError("invalid_part", "Embedding content parts must be mappings")
        part_type = part.get("type")
        if part_type == "text":
            if set(part) != {"type", "text"} or not isinstance(part.get("text"), str) or not part["text"]:
                raise EmbeddingInputError("invalid_text", "Text parts must contain nonempty text")
            text_bytes += len(part["text"].encode("utf-8"))
            continue
        if part_type not in ("image", "audio", "video"):
            raise EmbeddingInputError("unsupported_part", f"Unsupported embedding content part: {part_type!r}")
        expected_keys = {"type", "media"} if part_type == "image" else {"type", "media", "duration_seconds"}
        if set(part) != expected_keys:
            raise EmbeddingInputError("invalid_part", f"Unexpected fields on {part_type} part")
        size = _validate_media(part.get("media"), part_type)
        total_bytes += size
        if part_type == "image":
            images += 1
            image_bytes += size
            media_tokens += EMBEDDINGGEMMA2_LIMITS["image_tokens"]
        elif part_type == "audio":
            audio += 1
            _validate_duration(part.get("duration_seconds"), "audio")
        else:
            video += 1
            _validate_duration(part.get("duration_seconds"), "video")
            frames = math.ceil(
                part["duration_seconds"] * EMBEDDINGGEMMA2_LIMITS["video_frames_per_second"]
            )
            if frames > EMBEDDINGGEMMA2_LIMITS["max_video_frames"]:
                raise EmbeddingInputError("video_frame_limit_exceeded", "Video exceeds the 32-frame limit")
            media_tokens += frames * EMBEDDINGGEMMA2_LIMITS["video_frame_tokens"]
    if images > EMBEDDINGGEMMA2_LIMITS["max_images_per_item"] or image_bytes > EMBEDDINGGEMMA2_LIMITS["max_image_bytes_per_item"]:
        raise EmbeddingInputError("image_limit_exceeded", "An item exceeds the image count or byte limit")
    if audio > 1 or video > 1:
        raise EmbeddingInputError("media_count_exceeded", "Each item supports at most one audio and one video part")
    return total_bytes, text_bytes, media_tokens, audio > 0


def validate_embedding_request(
    model: str,
    input_value: EmbeddingInput,
    *,
    dimensions: Optional[int],
    input_type: Optional[str],
) -> None:
    """Validate SDK-visible EmbeddingGemma 2 constraints without changing wire order."""

    if input_type is not None and input_type not in ("query", "document", "unspecified"):
        raise EmbeddingInputError("invalid_input_type", "input_type must be query, document, or unspecified")
    if model != EMBEDDINGGEMMA2_MODEL:
        if input_type == "unspecified":
            raise EmbeddingInputError("invalid_input_type", "unspecified input_type is only supported by embeddinggemma-2")
        return
    if dimensions is not None and dimensions not in EMBEDDINGGEMMA2_DIMENSIONS:
        raise EmbeddingInputError(
            "invalid_dimensions",
            "embeddinggemma-2 dimensions must be one of 768, 512, 256, or 128",
        )
    items: Sequence[Any]
    if isinstance(input_value, str):
        if not input_value:
            raise EmbeddingInputError("invalid_input", "Embedding input must not be empty")
        items = [input_value]
    elif isinstance(input_value, Sequence) and not isinstance(input_value, (bytes, bytearray)):
        items = input_value
    else:
        raise EmbeddingInputError("invalid_input", "Embedding input must be a string or a list")
    max_items = EMBEDDINGGEMMA2_LIMITS["max_batch_items"]
    if not items or len(items) > max_items:
        raise EmbeddingInputError("invalid_batch", f"Embedding input requires 1-{max_items} items")
    request_media_bytes = total_text_bytes = 0
    prefix = "" if input_type == "unspecified" else "title: none | text: " if input_type == "document" else "task: search result | query: "
    for item in items:
        if isinstance(item, str):
            if not item:
                raise EmbeddingInputError("invalid_input", "Embedding text items must not be empty")
            text_bytes = len(item.encode("utf-8"))
            media_tokens = 0
        else:
            media_bytes, text_bytes, media_tokens, _has_audio = _validate_item(item)
            request_media_bytes += media_bytes
        total_text_bytes += text_bytes
        if text_bytes + len(prefix.encode("utf-8")) + EMBEDDINGGEMMA2_LIMITS["processor_template_tokens"] + media_tokens > EMBEDDINGGEMMA2_LIMITS["max_processed_tokens_per_item"]:
            raise EmbeddingInputError("context_limit_exceeded", "An item exceeds the 8192 processed-token admission bound")
    if total_text_bytes > EMBEDDINGGEMMA2_LIMITS["max_total_text_bytes"]:
        raise EmbeddingInputError("text_too_large", "Request exceeds the 10 MiB text limit")
    if request_media_bytes > EMBEDDINGGEMMA2_LIMITS["max_request_media_bytes"]:
        raise EmbeddingInputError("request_media_too_large", "Request exceeds 20 MiB of decoded media")


__all__ = [
    "EMBEDDINGGEMMA2_MODEL",
    "EMBEDDINGGEMMA2_DIMENSIONS",
    "EMBEDDINGGEMMA2_PROTOCOL",
    "EMBEDDINGGEMMA2_LIMITS",
    "EmbeddingInputError",
    "media_from_bytes",
    "media_from_base64",
    "media_from_file",
    "text_part",
    "image_part",
    "audio_part",
    "video_part",
    "multimodal_item",
]
