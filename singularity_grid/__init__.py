"""singularity-grid -- Python SDK for the SGL Network compute grid."""

from .processors import ProcessorsClient, PROCESSORS_BASE_URL
from .client import (
    DEFAULT_BASE_URL,
    GridClient,
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
    JobResponse,
    JobResult,
    ModelInfo,
    ModelPricing,
    ModelsResponse,
    PricingInfo,
    PricingResponse,
    TeeCapacity,
)
from .openai_compat import create_openai_client

__version__ = "0.2.0"

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
]
from .vault import VaultClient, VaultError, encrypt_envelope, decrypt_envelope
