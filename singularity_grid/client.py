"""GridClient — main entry point for the SGL Network compute grid."""

from __future__ import annotations

import json as _json
from typing import Any, Dict, Iterator, List, Optional

import httpx

from . import e2e
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

DEFAULT_BASE_URL = "https://grid.x402compute.cc"

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
            # Grid credit billing reads X-API-Key; send both so reserve + chat
            # resolve the paying wallet (credits mode) for end-to-end requests.
            headers["X-API-Key"] = api_key
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

    # -- chat (end-to-end encrypted) ---------------------------------------

    def _reserve(self, model: str) -> Dict[str, Any]:
        """Reserve a node + learn its X25519 key so we can seal the prompt to it."""
        data = self._request("POST", "/v1/reserve", json={"model": model})
        if not data.get("node_x25519_pubkey"):
            raise SGLAPIError(503, "Reserved node does not support E2E encryption")
        return data

    def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """End-to-end encrypted chat completion.

        The prompt is sealed in this client to the serving node's key and only
        decrypts inside its TEE — the orchestrator only relays ciphertext. Requires
        ``api_key`` (credits); x402 pay-per-call isn't signed here (use the wallet
        flow). Returns an OpenAI-style dict with an extra ``attestation`` field.
        """
        reservation = self._reserve(model)
        resp_sk, resp_pub = e2e.new_response_keypair()
        sealed_ct, eph = e2e.seal_input(
            reservation["node_x25519_pubkey"], resp_pub,
            _json.dumps({"messages": messages, "temperature": temperature, "max_tokens": max_tokens}).encode(),
        )
        body = {
            "reservation_token": reservation["reservation_token"],
            "max_tokens": max_tokens,  # cleartext, only used to quote the x402 price
            "enc": {
                "ciphertext": sealed_ct,
                "client_ephemeral_pubkey": eph,
                "client_response_pubkey": resp_pub,
                "algorithm": e2e.ALGO_V2,
            },
        }
        try:
            data = self._request("POST", "/v1/chat/completions", json=body)
        except SGLAPIError as err:
            if err.status_code == 402:
                raise SGLAPIError(402, "Payment required — pass api_key (credits); the Python GridClient does not sign x402 payments.") from err
            raise

        sealed = data.get("sealed_result")
        if not sealed:
            raise SGLAPIError(500, "No sealed result returned")
        plain = e2e.open_output(resp_sk, resp_pub, sealed["ephemeral_public_key"], sealed["ciphertext"])
        parsed = _json.loads(plain)
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": parsed.get("content", "")}, "finish_reason": "stop"}],
            "usage": data.get("usage") or parsed.get("usage", {}),
            "attestation": {
                "node_id": reservation.get("node_id"),
                "tee_type": reservation.get("tee_type"),
                "verified": bool(reservation.get("attestation_verified", False)),
            },
        }

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """Yield decoded text as it streams (end-to-end encrypted). Requires
        ``api_key`` (credits). Each chunk is decrypted and its ordering +
        termination verified (a truncated stream raises). If streaming isn't
        enabled server-side, the whole reply is yielded as one chunk."""
        reservation = self._reserve(model)
        resp_sk, resp_pub = e2e.new_response_keypair()
        nonce = e2e.random_nonce_b58()
        sealed_ct, eph = e2e.seal_input(
            reservation["node_x25519_pubkey"], resp_pub,
            _json.dumps({"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True, "nonce": nonce}).encode(),
        )
        body = {
            "reservation_token": reservation["reservation_token"],
            "stream": True,
            "max_tokens": max_tokens,
            "enc": {
                "ciphertext": sealed_ct,
                "client_ephemeral_pubkey": eph,
                "client_response_pubkey": resp_pub,
                "algorithm": e2e.ALGO_V2,
            },
        }
        with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                resp.read()
                if resp.status_code == 402:
                    raise SGLAPIError(402, "Payment required — pass api_key (credits); the Python GridClient does not sign x402 payments.")
                msg = resp.text
                try:
                    err = resp.json().get("error")
                    msg = err.get("message", msg) if isinstance(err, dict) else (err or msg)
                except Exception:
                    pass
                raise SGLAPIError(resp.status_code, str(msg))

            if "text/event-stream" not in resp.headers.get("content-type", ""):
                resp.read()
                data = _json.loads(resp.text)
                sealed = data.get("sealed_result")
                if not sealed:
                    raise SGLAPIError(500, "No sealed result returned")
                plain = e2e.open_output(resp_sk, resp_pub, sealed["ephemeral_public_key"], sealed["ciphertext"])
                content = _json.loads(plain).get("content", "")
                if content:
                    yield content
                return

            expected_seq = 0
            out_key = None
            stream_eph = None
            saw_final = False
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: error"):
                    raise SGLAPIError(502, "stream aborted by server")
                if line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunk = _json.loads(payload)
                except _json.JSONDecodeError as exc:
                    raise SGLAPIError(502, "malformed stream chunk") from exc
                seq = chunk.get("seq")
                if seq is None or "ct" not in chunk:
                    raise SGLAPIError(502, "invalid stream chunk (missing seq/ciphertext)")
                if seq != expected_seq:
                    raise SGLAPIError(502, f"stream out of order (expected {expected_seq}, got {seq})")
                if seq == 0:
                    stream_eph = chunk.get("eph")
                    if not stream_eph:
                        raise SGLAPIError(502, "stream chunk 0 missing ephemeral key")
                    out_key = e2e.stream_out_key(resp_sk, stream_eph)
                is_final = chunk.get("final") is True
                text = e2e.open_stream_chunk(out_key, resp_pub, stream_eph, nonce, seq, is_final, chunk["ct"]).decode()
                if text:
                    yield text
                expected_seq += 1
                if is_final:
                    saw_final = True
                    break
            if not saw_final:
                raise SGLAPIError(502, "stream ended before final chunk (truncated)")

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
