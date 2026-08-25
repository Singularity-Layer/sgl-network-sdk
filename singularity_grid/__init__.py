"""singularity-grid -- Python SDK for the SGL Network compute grid."""

from .client import (
    DEFAULT_BASE_URL,
    GridClient,
    SGLAPIError,
    SGLAuthError,
    SGLConnectionError,
    SGLError,
    SGLNotFoundError,
)
from .models import (
    AttestationProof,
    CapacityResponse,
    JobResponse,
    JobResult,
    ModelInfo,
    ModelPricing,
    ModelsResponse,
    PricingInfo,
    PricingResponse,
    ProcessorDeployResult,
    ProcessorInfo,
    ProcessorListResponse,
    ProcessorLogEntry,
    ProcessorLogsResponse,
    ProcessorPayment,
    ProcessorResult,
    ProcessorTeeInfo,
    TeeCapacity,
    TrainArtifacts,
    TrainDatasetApproval,
    TrainDatasetInfo,
    TrainRunCreated,
    TrainRunInfo,
)
from .openai_compat import create_openai_client
from .train import TrainAPI

__version__ = "0.7.0"

__all__ = [
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
    # Models — Grid
    "AttestationProof",
    "CapacityResponse",
    "JobResponse",
    "JobResult",
    "ModelInfo",
    "ModelPricing",
    "ModelsResponse",
    "PricingInfo",
    "PricingResponse",
    "TeeCapacity",
    # Models — Processors
    "ProcessorDeployResult",
    "ProcessorInfo",
    "ProcessorListResponse",
    "ProcessorLogEntry",
    "ProcessorLogsResponse",
    "ProcessorPayment",
    "ProcessorResult",
    "ProcessorTeeInfo",
    # Managed training
    "TrainAPI",
    "TrainArtifacts",
    "TrainDatasetApproval",
    "TrainDatasetInfo",
    "TrainRunCreated",
    "TrainRunInfo",
]
