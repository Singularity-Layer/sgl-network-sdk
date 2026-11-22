"""Stable exception types shared by the SGL Network SDK clients and helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


class SGLError(Exception):
    """Base exception for all SGL Network SDK errors."""


class EmbeddingInputError(SGLError, ValueError):
    """Local embedding validation error raised before a request is sent."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class TranscriptionInputError(SGLError, ValueError):
    """Local transcription validation error raised before audio is sent."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class TranscriptionResponseError(SGLError):
    """A signed transcription envelope, result, or request binding was invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class SGLAPIError(SGLError):
    """Raised when the API returns a non-2xx status code."""

    def __init__(
        self,
        status_code: int,
        message: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        # x402 response headers, when the server sent any (PAYMENT-REQUIRED on a 402 carries the
        # v2 offers a standard x402 client signs). Keys are the header names as sent.
        self.headers: Dict[str, str] = dict(headers or {})
        error = body.get("error") if isinstance(body, dict) else None
        self.code = (
            error.get("code") or error.get("type")
            if isinstance(error, dict)
            else body.get("code") if isinstance(body, dict) else None
        )
        super().__init__(f"HTTP {status_code}: {message}")

    @property
    def payment_required(self) -> Optional[str]:
        """The raw ``PAYMENT-REQUIRED`` header (base64 x402 v2 offers), if the server sent one."""
        return self.headers.get("PAYMENT-REQUIRED")


class SGLAuthError(SGLAPIError):
    """Raised on 401/403 responses."""


class SGLNotFoundError(SGLAPIError):
    """Raised on 404 responses."""


class SGLConnectionError(SGLError):
    """Raised when the orchestrator is unreachable."""
