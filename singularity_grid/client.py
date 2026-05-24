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

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "GridClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
