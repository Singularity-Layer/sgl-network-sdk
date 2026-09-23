"""Pydantic models for SGL Network API responses."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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
# System One (typed decisions, e.g. Laya/Jev)
# ---------------------------------------------------------------------------

SystemOneTier = Literal["standard", "confidential"]


class SystemOneChoiceQuestion(BaseModel):
    """Pick one of ``criteria``: a mapping of option id to description (at least two)."""
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Dict[str, str]


class SystemOneScoreQuestion(BaseModel):
    """Score the state against a non-empty list of criteria."""
    type: Literal["score"] = "score"
    instructions: str
    criteria: List[str]


class SystemOneNoulQuestion(BaseModel):
    """Free-form typed value (string, number or boolean). Takes no criteria."""
    type: Literal["noul"] = "noul"
    instructions: str


SystemOneQuestion = Annotated[
    Union[SystemOneChoiceQuestion, SystemOneScoreQuestion, SystemOneNoulQuestion],
    Field(discriminator="type"),
]


class SystemOneChoiceAnswer(BaseModel):
    """Answer to a ``choice`` question."""
    model_config = ConfigDict(extra="allow")
    type: Literal["choice"]
    choice: str
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


class SystemOneScoreAnswer(BaseModel):
    """Answer to a ``score`` question."""
    model_config = ConfigDict(extra="allow")
    type: Literal["score"]
    score: float
    confidence: Optional[float] = None


class SystemOneNoulAnswer(BaseModel):
    """Answer to a ``noul`` question."""
    model_config = ConfigDict(extra="allow")
    type: Literal["noul"]
    value: Optional[Union[bool, int, float, str]] = None
    confidence: Optional[float] = None


SystemOneAnswer = Annotated[
    Union[SystemOneChoiceAnswer, SystemOneScoreAnswer, SystemOneNoulAnswer],
    Field(discriminator="type"),
]


class SystemOneUsage(BaseModel):
    """Usage for a System One call. Billed on input tokens only."""
    model_config = ConfigDict(extra="allow")
    input_tokens: int = 0
    output_tokens: Union[int, float] = 0
    cost_usd: float = 0.0


class SystemOneResponse(BaseModel):
    """Response from POST /v1/systemone. Unknown node fields are kept as extras."""
    model_config = ConfigDict(extra="allow")
    object: Literal["systemone.result"] = "systemone.result"
    model: str
    answers: Dict[str, SystemOneAnswer]
    usage: SystemOneUsage = Field(default_factory=SystemOneUsage)


class SystemOneModelInfo(BaseModel):
    """System One model descriptor returned by GET /v1/models?type=systemone."""
    model_config = ConfigDict(extra="allow")
    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: str = ""
    type: Literal["systemone"] = "systemone"
    context_window: Optional[int] = None
    max_questions: Optional[int] = None
    degraded: bool = False
