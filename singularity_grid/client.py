"""GridClient — main entry point for the SGL Network compute grid."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from .models import (
    AttestationProof,
    CapacityResponse,
    JobResponse,
    JobResult,
    ModelInfo,
    ModelsResponse,
    PricingInfo,
    PricingResponse,
    ProcessorDeployResult,
    ProcessorInfo,
    ProcessorListResponse,
    ProcessorLogEntry,
    ProcessorLogsResponse,
    ProcessorResult,
)

DEFAULT_BASE_URL = (
    "https://sgl-network-orchestrator.ivaavimusicproductions.workers.dev"
)

DEFAULT_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SGLError(Exception):
    """Base exception for all SGL Network SDK errors."""


class SGLAPIError(SGLError):
    """Raised when the API returns a non-2xx status code."""

    def __init__(
        self,
        status_code: int,
        message: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {message}")


class SGLAuthError(SGLAPIError):
    """Raised on 401/403 responses."""


class SGLNotFoundError(SGLAPIError):
    """Raised on 404 responses."""


class SGLConnectionError(SGLError):
    """Raised when the orchestrator is unreachable."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GridClient:
    """Client for the SGL Network orchestrator API.

    Parameters
    ----------
    api_key:
        Bearer token for authenticated endpoints (e.g. ``scg_...``).
        Not required for public endpoints like ``capacity()``.
    base_url:
        Override the default orchestrator URL.
    timeout:
        Request timeout in seconds (default 60).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        headers: Dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )

    # -- helpers ------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request and return the parsed JSON body."""
        try:
            response = self._client.request(
                method, path, json=json, params=params
            )
        except httpx.ConnectError as exc:
            raise SGLConnectionError(
                f"Could not connect to {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise SGLConnectionError(
                f"Request to {self._base_url}{path} timed out: {exc}"
            ) from exc

        if response.status_code >= 400:
            body: Optional[Dict[str, Any]] = None
            message = response.text
            try:
                body = response.json()
                message = body.get("error", {}).get("message", message) if isinstance(body.get("error"), dict) else body.get("error", message)
            except Exception:
                pass

            if response.status_code in (401, 403):
                raise SGLAuthError(response.status_code, str(message), body)
            if response.status_code == 404:
                raise SGLNotFoundError(response.status_code, str(message), body)
            raise SGLAPIError(response.status_code, str(message), body)

        if response.status_code == 204:
            return {}
        return response.json()  # type: ignore[no-any-return]

    # -- public endpoints (no auth required) --------------------------------

    def capacity(self) -> CapacityResponse:
        """Return grid-wide capacity summary."""
        data = self._request("GET", "/grid/capacity")
        return CapacityResponse.model_validate(data)

    def models(self) -> List[ModelInfo]:
        """Return available models with pricing and node info."""
        data = self._request("GET", "/grid/models")
        wrapped = ModelsResponse.model_validate(data)
        return wrapped.models

    def pricing(self) -> List[PricingInfo]:
        """Return the pricing table for all models."""
        data = self._request("GET", "/grid/pricing")
        wrapped = PricingResponse.model_validate(data)
        return wrapped.pricing

    # -- authenticated endpoints --------------------------------------------

    def submit_job(
        self,
        model: str,
        input_payload: Dict[str, Any],
        *,
        submitter_wallet: Optional[str] = None,
        submitter_chain: Optional[str] = None,
    ) -> JobResponse:
        """Submit a compute job to the grid.

        Parameters
        ----------
        model:
            Model identifier (e.g. ``"gemma-4-26b"``).
        input_payload:
            The request body for the model (e.g. chat messages).
        submitter_wallet:
            On-chain wallet address of the submitter.
        submitter_chain:
            Chain identifier (e.g. ``"base"``, ``"ethereum"``).
        """
        body: Dict[str, Any] = {
            "model": model,
            "input": input_payload,
        }
        if submitter_wallet is not None:
            body["submitter_wallet"] = submitter_wallet
        if submitter_chain is not None:
            body["submitter_chain"] = submitter_chain

        data = self._request("POST", "/grid/jobs", json=body)
        return JobResponse.model_validate(data)

    def get_job(self, job_id: str) -> JobResult:
        """Get the status and result of a previously submitted job."""
        data = self._request("GET", f"/grid/jobs/{job_id}")
        return JobResult.model_validate(data)

    def get_attestation(self, job_id: str) -> AttestationProof:
        """Retrieve the TEE attestation proof for a completed job."""
        data = self._request("GET", f"/grid/jobs/{job_id}/attestation")
        return AttestationProof.model_validate(data)

    # -- processor endpoints -----------------------------------------------

    def deploy_processor(
        self,
        name: str,
        code: str,
        *,
        wallet_address: str,
        chain: str = "solana",
        runtime: str = "deno",
        memory_mb: int = 128,
        timeout_seconds: int = 30,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProcessorDeployResult:
        """Deploy a serverless processor to the TEE grid.

        Requires $SGL staking for higher tier limits.

        Parameters
        ----------
        name:
            Processor name (lowercase alphanumeric + hyphens, 2-64 chars).
        code:
            JavaScript/TypeScript source code with a default export function.
        wallet_address:
            Owner wallet address (must have $SGL staked for higher tiers).
        chain:
            Chain the wallet is on (default ``"solana"``).
        runtime:
            Execution runtime (``"deno"`` or ``"wasm"``).
        memory_mb:
            Memory allocation in MB (64, 128, 256, 512, 1024).
        timeout_seconds:
            Max execution time (1-300s, tier-dependent).
        metadata:
            Optional metadata dict attached to the processor.
        """
        body: Dict[str, Any] = {
            "name": name,
            "code": code,
            "runtime": runtime,
            "memory_mb": memory_mb,
            "timeout_seconds": timeout_seconds,
        }
        if metadata:
            body["metadata"] = metadata

        # Set wallet auth headers
        self._client.headers["X-Auth-Address"] = wallet_address
        self._client.headers["X-Auth-Chain"] = chain
        try:
            data = self._request("POST", "/grid/processors", json=body)
        finally:
            self._client.headers.pop("X-Auth-Address", None)
            self._client.headers.pop("X-Auth-Chain", None)

        return ProcessorDeployResult.model_validate(data)

    def invoke_processor(
        self,
        name: str,
        input_data: Dict[str, Any],
        *,
        payment_header: str,
        payment_token: str = "USDC",
    ) -> ProcessorResult:
        """Invoke a deployed processor with x402 payment.

        Parameters
        ----------
        name:
            Processor name.
        input_data:
            Input object passed to the handler function.
        payment_header:
            x402 payment header (JSON-encoded).
        payment_token:
            Payment token type (``"USDC"`` or ``"SGL"``).
        """
        body: Dict[str, Any] = {
            "input": input_data,
            "payment_token": payment_token,
        }

        self._client.headers["X-Payment"] = payment_header
        try:
            data = self._request(
                "POST", f"/grid/processors/{name}/invoke", json=body,
            )
        finally:
            self._client.headers.pop("X-Payment", None)

        return ProcessorResult.model_validate(data)

    def list_processors(
        self,
        *,
        owner: Optional[str] = None,
        page: int = 0,
        limit: int = 50,
    ) -> ProcessorListResponse:
        """List deployed processors."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        if owner:
            params["owner"] = owner
        data = self._request("GET", "/grid/processors", params=params)
        return ProcessorListResponse.model_validate(data)

    def get_processor(self, processor_id: str) -> ProcessorInfo:
        """Get details for a specific processor."""
        data = self._request("GET", f"/grid/processors/{processor_id}")
        return ProcessorInfo.model_validate(data)

    def delete_processor(
        self,
        processor_id: str,
        *,
        wallet_address: str,
    ) -> Dict[str, Any]:
        """Delete a processor (must be the owner)."""
        self._client.headers["X-Auth-Address"] = wallet_address
        try:
            data = self._request(
                "DELETE", f"/grid/processors/{processor_id}",
            )
        finally:
            self._client.headers.pop("X-Auth-Address", None)
        return data

    def get_processor_logs(
        self,
        processor_id: str,
        *,
        page: int = 0,
        limit: int = 50,
    ) -> ProcessorLogsResponse:
        """Get invocation logs for a processor."""
        params: Dict[str, Any] = {"page": page, "limit": limit}
        data = self._request(
            "GET",
            f"/grid/processors/{processor_id}/logs",
            params=params,
        )
        return ProcessorLogsResponse.model_validate(data)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "GridClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
