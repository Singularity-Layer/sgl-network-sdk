"""singularity-grid -- Python SDK for the SGL Network compute grid."""

from .processors import ProcessorsClient, PROCESSORS_BASE_URL
from .client import (
    DEFAULT_BASE_URL,
    GridClient,
)
from .errors import (
    EmbeddingInputError,
    SGLAPIError,
    SGLAuthError,
    SGLConnectionError,
    SGLError,
    SGLNotFoundError,
)
from .pods import (
    DEFAULT_PODS_BASE_URL,
    PodsClient,
    verify_pod_webhook,
)
from .models import (
    AttestationProof,
    CapacityResponse,
    EmbeddingAudioPart,
    EmbeddingAudioMimeType,
    EmbeddingContentPart,
    EmbeddingData,
    EmbeddingDimension,
    EmbeddingImagePart,
    EmbeddingImageMimeType,
    EmbeddingInput,
    EmbeddingInputItem,
    EmbeddingInputType,
    EmbeddingModality,
    EmbeddingMediaMimeType,
    EmbeddingResponse,
    EmbeddingTextPart,
    EmbeddingUsage,
    EmbeddingUsageBreakdown,
    EmbeddingVideoPart,
    EmbeddingVideoMimeType,
    InlineEmbeddingMedia,
    JobResponse,
    JobResult,
    ModelInfo,
    ModelPricing,
    ModelsResponse,
    PricingInfo,
    PricingResponse,
    TeeCapacity,
    MultimodalEmbeddingItem,
)
from .embeddings import (
    EMBEDDINGGEMMA2_DIMENSIONS,
    EMBEDDINGGEMMA2_LIMITS,
    EMBEDDINGGEMMA2_MIME_TYPES,
    EMBEDDINGGEMMA2_MODEL,
    EMBEDDINGGEMMA2_PROTOCOL,
    audio_part,
    image_part,
    media_from_base64,
    media_from_bytes,
    media_from_file,
    multimodal_item,
    text_part,
    video_part,
)
from .openai_compat import create_openai_client

__version__ = "0.11.0"

__all__ = [
    # Processors — a SEPARATE client on processors.x402compute.cc. Until 0.9.0 these methods
    # lived on GridClient and pointed at /grid/processors, which has never existed.
    "ProcessorsClient",
    "PROCESSORS_BASE_URL",
    # Agent Pods — hosted agents, on compute.x402layer.cc.
    "PodsClient",
    "DEFAULT_PODS_BASE_URL",
    "verify_pod_webhook",
    # Client
    "GridClient",
    "create_openai_client",
    "DEFAULT_BASE_URL",
    # Exceptions
    "SGLError",
    "SGLAPIError",
    "SGLAuthError",
    "SGLConnectionError",
    "SGLNotFoundError",
    "EmbeddingInputError",
    # EmbeddingGemma 2 constants + safe media/input helpers
    "EMBEDDINGGEMMA2_MODEL",
    "EMBEDDINGGEMMA2_DIMENSIONS",
    "EMBEDDINGGEMMA2_PROTOCOL",
    "EMBEDDINGGEMMA2_LIMITS",
    "EMBEDDINGGEMMA2_MIME_TYPES",
    "media_from_bytes",
    "media_from_base64",
    "media_from_file",
    "text_part",
    "image_part",
    "audio_part",
    "video_part",
    "multimodal_item",
    # Models — Grid
    "AttestationProof",
    "CapacityResponse",
    "EmbeddingAudioPart",
    "EmbeddingAudioMimeType",
    "EmbeddingContentPart",
    "EmbeddingData",
    "EmbeddingDimension",
    "EmbeddingImagePart",
    "EmbeddingImageMimeType",
    "EmbeddingInput",
    "EmbeddingInputItem",
    "EmbeddingInputType",
    "EmbeddingModality",
    "EmbeddingMediaMimeType",
    "EmbeddingResponse",
    "EmbeddingTextPart",
    "EmbeddingUsage",
    "EmbeddingUsageBreakdown",
    "EmbeddingVideoPart",
    "EmbeddingVideoMimeType",
    "InlineEmbeddingMedia",
    "MultimodalEmbeddingItem",
    "JobResponse",
    "JobResult",
    "ModelInfo",
    "ModelPricing",
    "ModelsResponse",
    "PricingInfo",
    "PricingResponse",
    "TeeCapacity",
]
from .vault import VaultClient, VaultError, encrypt_envelope, decrypt_envelope
