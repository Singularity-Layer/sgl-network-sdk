"""Helper to configure the OpenAI SDK for use with the SGL Network."""

from __future__ import annotations

from typing import Optional

from .client import DEFAULT_BASE_URL


def create_openai_client(
    api_key: str = "sgl-anonymous",
    base_url: Optional[str] = None,
    **kwargs,
):
    """Return an ``openai.OpenAI`` instance pointed at the SGL Network.

    Parameters
    ----------
    api_key:
        API key for the orchestrator (e.g. ``"scg_..."``).
        Defaults to ``"sgl-anonymous"`` for unauthenticated access.
    base_url:
        Override the default orchestrator URL. The ``/v1`` suffix is
        appended automatically if not already present.
    **kwargs:
        Additional keyword arguments forwarded to ``openai.OpenAI()``.

    Returns
    -------
    openai.OpenAI
        A configured OpenAI client.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed. Install it with
        ``pip install singularity-grid[openai]``.
    """
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The openai package is required for create_openai_client(). "
            "Install it with: pip install singularity-grid[openai]"
        ) from None

    url = base_url or DEFAULT_BASE_URL
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"

    return OpenAI(api_key=api_key, base_url=url, **kwargs)
