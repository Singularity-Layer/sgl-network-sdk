"""Pydantic models for SGL Network API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

class TeeCapacity(BaseModel):
    """Capacity breakdown for a single TEE type."""
    tee_type: str
    total_nodes: int = 0
    active_nodes: int = 0
    available_nodes: int = 0


class CapacityResponse(BaseModel):
    """Grid-wide capacity summary returned by GET /grid/capacity."""
    total_nodes: int = 0
    active_nodes: int = 0
    available_nodes: int = 0
    by_tee_type: List[TeeCapacity] = Field(default_factory=list)
    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelPricing(BaseModel):
    """Pricing for a specific model."""
    price_per_1k_input_tokens_usd: float = 0.0
    price_per_1k_output_tokens_usd: float = 0.0


class ModelInfo(BaseModel):
    """Model descriptor returned by GET /grid/models."""
    id: str
    owned_by: str = ""
    sgl_node_count: int = 0
    sgl_tee_types: List[str] = Field(default_factory=list)
    sgl_pricing: Optional[ModelPricing] = None


class ModelsResponse(BaseModel):
    """Wrapper for the models list."""
    models: List[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class PricingInfo(BaseModel):
    """Pricing entry for a single model."""
    model: str
    price_per_1k_input_tokens_usd: float = 0.0
    price_per_1k_output_tokens_usd: float = 0.0


class PricingResponse(BaseModel):
    """Full pricing table returned by GET /grid/pricing."""
    pricing: List[PricingInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    """Response from POST /grid/jobs (job submission)."""
    job_id: str
    status: str = "pending"
    model: str = ""
    node_id: Optional[str] = None
    tee_type: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    created_at: Optional[str] = None


class AttestationProof(BaseModel):
    """TEE attestation proof for a completed job."""
    node_id: str = ""
    tee_type: str = ""
    job_id: str = ""
    attestation_signature: str = ""
    attestation_report: Optional[str] = None
    verified: bool = False
    verified_at: Optional[str] = None


class JobResult(BaseModel):
    """Full job result returned by GET /grid/jobs/:id."""
    id: str
    status: str = "pending"
    model: str = ""
    node_id: Optional[str] = None
    tee_type: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    encrypted_result: Optional[str] = None
    attestation_proof: Optional[AttestationProof] = None
    cost_usd: Optional[float] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EmbeddingModality = Literal["text", "image", "audio", "video"]
EmbeddingInputType = Literal["query", "document", "unspecified"]
EmbeddingDimension = Literal[768, 512, 256, 128]


class InlineEmbeddingMedia(TypedDict):
    """Canonical media envelope accepted by EmbeddingGemma 2."""

    encoding: Literal["base64"]
    mime_type: str
    data: str
    sha256: str


class EmbeddingTextPart(TypedDict):
    type: Literal["text"]
    text: str


class EmbeddingImagePart(TypedDict):
    type: Literal["image"]
    media: InlineEmbeddingMedia


class EmbeddingAudioPart(TypedDict):
    type: Literal["audio"]
    media: InlineEmbeddingMedia
    duration_seconds: float


class EmbeddingVideoPart(TypedDict):
    type: Literal["video"]
    media: InlineEmbeddingMedia
    duration_seconds: float


EmbeddingContentPart = Union[
    EmbeddingTextPart,
    EmbeddingImagePart,
    EmbeddingAudioPart,
    EmbeddingVideoPart,
]


class MultimodalEmbeddingItem(TypedDict):
    """One vector input made from ordered content parts."""

    content: List[EmbeddingContentPart]


EmbeddingInputItem = Union[str, MultimodalEmbeddingItem]
EmbeddingInput = Union[str, Sequence[EmbeddingInputItem]]


class EmbeddingData(TypedDict):
    object: str
    index: int
    embedding: List[float]


class EmbeddingUsageBreakdown(TypedDict):
    text: int
    image: int
    audio: int
    video: int


class EmbeddingUsage(TypedDict, total=False):
    prompt_tokens: int
    total_tokens: int
    cost_usd: float
    breakdown: EmbeddingUsageBreakdown


class EmbeddingResponse(TypedDict, total=False):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage
    processor_revision: str
    embedding_protocol: str


# ---------------------------------------------------------------------------
