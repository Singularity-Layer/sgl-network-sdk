"""Pydantic models for SGL Network API responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
# Processors
# ---------------------------------------------------------------------------

class ProcessorInfo(BaseModel):
    """Processor descriptor returned by list/get endpoints."""
    id: str
    name: str
    owner_wallet: str = ""
    owner_chain: str = "solana"
    runtime: str = "deno"
    memory_mb: int = 128
    timeout_seconds: int = 30
    invocation_count: int = 0
    last_invoked_at: Optional[str] = None
    status: str = "active"
    deployment_stake_sgl: float = 0.0
    invoke_url: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessorDeployResult(BaseModel):
    """Response from deploying a processor."""
    id: str
    name: str
    runtime: str = "deno"
    memory_mb: int = 128
    timeout_seconds: int = 30
    status: str = "active"
    code_hash: str = ""
    invoke_url: str = ""
    tier: str = "free"
    sgl_staked: float = 0.0
    created_at: Optional[str] = None


class ProcessorPayment(BaseModel):
    """Payment details from a processor invocation."""
    amount_usd: str = "0"
    token: str = "USDC"
    discount_pct: float = 0.0


class ProcessorTeeInfo(BaseModel):
    """TEE details from a processor invocation."""
    node_id: str = ""
    tee_type: str = ""
    attestation_verified: bool = False


class ProcessorResult(BaseModel):
    """Response from invoking a processor."""
    job_id: str
    processor: str = ""
    status: str = "pending"
    output: Optional[Any] = None
    duration_ms: Optional[int] = None
    payment: Optional[ProcessorPayment] = None
    tee: Optional[ProcessorTeeInfo] = None


class ProcessorListResponse(BaseModel):
    """Paginated processor listing."""
    processors: List[ProcessorInfo] = Field(default_factory=list)
    total: int = 0
    page: int = 0
    limit: int = 50


class ProcessorLogEntry(BaseModel):
    """Single invocation log entry."""
    id: str = ""
    processor_id: str = ""
    job_id: str = ""
    node_id: Optional[str] = None
    duration_ms: Optional[int] = None
    status: str = ""
    error_message: Optional[str] = None
    created_at: Optional[str] = None


class ProcessorLogsResponse(BaseModel):
    """Paginated processor logs."""
    logs: List[ProcessorLogEntry] = Field(default_factory=list)
    total: int = 0
    page: int = 0
    limit: int = 50


# ─── Managed training ──────────────────────────────────────────────────────────


class TrainDatasetInfo(BaseModel):
    """A training dataset row (GET /training/datasets/:id)."""
    id: str = ""
    source: str = ""                       # 'upload' | 'synth'
    row_count: Optional[int] = None
    bytes: Optional[int] = None
    schema_ok: bool = False
    approved_at: Optional[str] = None
    created_at: str = ""
    synth_status: Optional[str] = None     # unpaid|pending|generating|complete|failed


class TrainDatasetApproval(BaseModel):
    """Result of the approval gate (POST /training/datasets/:id/approve)."""
    dataset_id: str = ""
    approved_at: str = ""
    row_count: int = 0
    total_rows: int = 0
    dropped: int = 0
    reasons: Dict[str, int] = Field(default_factory=dict)


class TrainRunInfo(BaseModel):
    """A training run (GET /training/runs/:id). `result` carries the runner report
    (baseline/final eval_loss+perplexity, improved, gguf_ok) once completed."""
    id: str = ""
    kind: str = ""
    status: str = ""
    outcome: str = ""
    base_model: str = ""
    dataset_id: str = ""
    gpu_plan_id: str = ""
    budget_hours: float = 0
    budget_credits: Optional[float] = None
    deadline_at: Optional[str] = None
    provisioned_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    exit_code: Optional[int] = None
    progress: Optional[Dict[str, Any]] = None   # {phase, pct: 0..1|None, at}
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class TrainRunCreated(BaseModel):
    """201 from POST /training/runs."""
    run_id: str = ""
    status: str = ""
    charged: float = 0
    billed_hours: float = 0
    gpu: str = ""
    base_model: str = ""
    note: str = ""


class TrainArtifacts(BaseModel):
    """Presigned downloads (POST /training/runs/:id/deploy target=download)."""
    run_id: str = ""
    target: str = "download"
    urls: Dict[str, str] = Field(default_factory=dict)   # adapter, report, gguf?
    expires_in: int = 0
    gguf_available: bool = False
