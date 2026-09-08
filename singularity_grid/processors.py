"""Singularity Processors.

A processor is a function we host: deploy one and you get a paid HTTP endpoint, an OpenAPI
document and an MCP server. Buyers pay the PUBLISHER directly in USDC over x402 — the platform
never holds it and takes no cut. The publisher pays for compute instead.

WHY THIS IS A SEPARATE CLIENT
-----------------------------
Processors live on their own worker at ``https://processors.x402compute.cc``, not on the grid.
Until 0.9.0 these methods hung off ``GridClient`` and pointed at ``/grid/processors``, which has
never existed — every call 404'd. Fixing that by teaching ``GridClient`` a second base URL would
have meant a per-call host override inside the shared ``_request`` that chat, embeddings and jobs
all use, which is a real risk to working features for no benefit. A separate client with its own
``httpx.Client`` touches none of that.

AUTH
----
A compute API key (``x402c_…``) with the ``processors:write`` scope, sent as ``X-API-Key``.

``processors:write`` is FULL CONTROL of processors owned by that key's wallet, delete and secrets
included — the same shape as a Cloudflare API token. If you want a credential that cannot change
anything, mint ``processors:read``. Three things to know before relying on it: compute keys do not
expire, there is no audit log of what a key did, and delete is permanent — the code is wiped and
the slug is burned forever, so a leaked key can destroy a name you can never reclaim.

Two routes never accept a key: ``suspend`` (moderation) and ``auth-session``. Both need a wallet
signature, and neither is a publisher action.

RUNNING A PROCESSOR IS NOT DONE WITH THE API KEY
------------------------------------------------
:meth:`ProcessorsClient.run` takes the INVOKE TOKEN that :meth:`deploy` returns, because the run
route is the only one with both a money path and an anonymous buyer lane and deliberately does not
read an API key as an ownership claim. Anonymous buyers pay with x402 instead; see
:meth:`run_with_payment`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse

import httpx

# The error hierarchy lives in client.py, not a separate errors module. Importing it here is
# one-directional: client.py does not import this file, so there is no cycle.
from .client import (
    SGLAPIError,
    SGLAuthError,
    SGLConnectionError,
    SGLNotFoundError,
)

PROCESSORS_BASE_URL = "https://processors.x402compute.cc"
DEFAULT_TIMEOUT = 60.0


def _is_loopback(hostname: str) -> bool:
    # urlparse strips the brackets from an IPv6 literal, so "::1" is what appears here.
    return hostname in ("localhost", "127.0.0.1", "::1", "[::1]")


def _assert_safe_base_url(raw: str) -> str:
    """A base URL override must not become a way to post the management key somewhere else.

    The key is long-lived, does not expire, and grants full control of the caller's processors,
    so an ``http://`` or attacker-supplied origin is a credential disclosure rather than a
    misconfiguration. Plain HTTP is allowed only for loopback, which is how you point this at a
    local worker during development.
    """
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"ProcessorsClient base_url is not a valid URL: {raw}")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and _is_loopback(parsed.hostname)):
        raise ValueError(
            f"ProcessorsClient base_url must be https (or http on localhost); got "
            f"{parsed.scheme}://{parsed.hostname}. The API key is a long-lived full-control "
            "credential and must not be sent in the clear."
        )
    return raw.rstrip("/")


def _key_allowed_on_host(raw: str) -> bool:
    """May the management key be sent to this host?

    https alone is not the question. ``https://attacker.example`` is a perfectly valid TLS
    origin, and if ``base_url`` is ever wired to an environment variable — which is exactly how
    people configure a staging host — then influencing that variable is enough to harvest a
    long-lived full-control credential. So the key travels only to the official host or to
    loopback unless the caller says otherwise, in one explicit flag they cannot set by accident.
    """
    def origin(u: str):
        p = urlparse(u)
        # Compare EFFECTIVE ports, so https://host and https://host:443 are the same origin.
        # Without this the official host written with its default port was refused.
        default = {"https": 443, "http": 80}.get(p.scheme or "")
        # `p.port or default` would fold an explicit :0 into the scheme default and
        # authorize https://host:0 as the official origin.
        return (p.scheme, p.hostname, p.port if p.port is not None else default)

    if origin(raw) == origin(PROCESSORS_BASE_URL):
        return True
    return _is_loopback(urlparse(raw).hostname or "")

__all__ = ["ProcessorsClient", "PROCESSORS_BASE_URL"]


class ProcessorsClient:
    """Manage Singularity Processors with a compute API key.

    Args:
        api_key: Compute API key (``x402c_…``) holding ``processors:read`` or
            ``processors:write``. Read-only calls to the public catalogue need none.
        base_url: Override the processors host. Must be https, or http on loopback.
        timeout: Per-request timeout in seconds.
        allow_key_on_custom_host: Send the API key to a ``base_url`` that is neither the official
            host nor loopback. Off by default; see :func:`_key_allowed_on_host`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = PROCESSORS_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        allow_key_on_custom_host: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = _assert_safe_base_url(base_url)
        headers: Dict[str, str] = {"Accept": "application/json"}
        if api_key:
            if not _key_allowed_on_host(self._base_url) and not allow_key_on_custom_host:
                raise ValueError(
                    f"ProcessorsClient refuses to send an API key to {self._base_url}. It is "
                    f"neither {PROCESSORS_BASE_URL} nor loopback. If you really do run your own "
                    "processors host, pass allow_key_on_custom_host=True."
                )
            headers["X-API-Key"] = api_key
        # trust_env=False for a LOOPBACK base URL, and only there.
        #
        # httpx honours HTTP_PROXY by default. With an http:// loopback host and a proxy set but
        # no matching NO_PROXY, the plaintext X-API-Key would be sent to that proxy — which
        # defeats the whole point of allowing plain HTTP on loopback in the first place. Against
        # the official https host the key is inside TLS, so a corporate proxy stays supported.
        loopback = _is_loopback(urlparse(self._base_url).hostname or "")
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
            trust_env=not loopback,
        )

    # -- helpers ------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        send_api_key: bool = True,
    ) -> Any:
        # Routes that carry their OWN credential — the public catalogue, and the two run paths,
        # which authenticate with an invoke token or an x402 payment — pass send_api_key=False,
        # so a long-lived management key is not scattered through request logs and traces on
        # calls that have no use for it.
        # httpx merges the client's default headers at build time, so the key has to be popped
        # off the built request — passing an override in `headers` would still send it.
        try:
            if send_api_key or not self._api_key:
                response = self._client.request(method, path, json=json, headers=headers)
            else:
                request = self._client.build_request(method, path, json=json, headers=headers)
                request.headers.pop("X-API-Key", None)
                response = self._client.send(request)
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
                if isinstance(body, dict):
                    # The worker answers {"error": ..., "detail": ...}; `detail` is the
                    # human sentence when it is present.
                    message = body.get("detail") or body.get("error") or message
            except Exception:
                pass

            if response.status_code in (401, 403):
                raise SGLAuthError(response.status_code, str(message), body)
            # 404 is "not yours" as well as "no such slug", deliberately: confirming which
            # private slugs exist to a non-owner is what the rest of the surface refuses to do.
            if response.status_code == 404:
                raise SGLNotFoundError(response.status_code, str(message), body)
            raise SGLAPIError(response.status_code, str(message), body)

        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ProcessorsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- discovery ----------------------------------------------------------

    def catalogue(self) -> Dict[str, Any]:
        """The public catalogue.

        Sends NO credential, even when the client holds one. ``GET /processors`` is owner-scoped
        when a key is presented and public otherwise, so passing the key here would silently
        return your own processors instead of the catalogue — the opposite of what the name
        promises. Use :meth:`list` when you want yours.
        """
        return self._request("GET", "/processors", send_api_key=False)

    def list(self) -> Dict[str, Any]:
        """Processors owned by this key's wallet. Needs ``processors:read``."""
        return self._request("GET", "/processors")

    def get(self, slug: str) -> Dict[str, Any]:
        """Owner projection when the key owns it, public projection otherwise."""
        return self._request("GET", f"/processors/{slug}")

    # -- lifecycle (needs processors:write) ---------------------------------

    def deploy(
        self,
        manifest: Dict[str, Any],
        code: Optional[str] = None,
        *,
        bundle: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Deploy a processor.

        The wallet behind the key becomes ``owner_wallet``, which is also the x402 ``payTo`` and
        the runtime-billing account — so the key must be minted on a SOLANA wallet, or this
        returns ``400 solana_wallet_required``.

        The ``invoke_token`` in the response is shown ONCE and never again.
        """
        body: Dict[str, Any] = {"manifest": manifest}
        if code is not None:
            body["code"] = code
        if bundle is not None:
            body["bundle"] = bundle
        if files is not None:
            body["files"] = files
        return self._request("POST", "/processors", json=body)

    def update(
        self,
        slug: str,
        *,
        manifest: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None,
        bundle: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Push new code, a new manifest, or both. Omitting ``manifest`` keeps the stored one."""
        body: Dict[str, Any] = {}
        if manifest is not None:
            body["manifest"] = manifest
        if code is not None:
            body["code"] = code
        if bundle is not None:
            body["bundle"] = bundle
        if files is not None:
            body["files"] = files
        return self._request("PATCH", f"/processors/{slug}", json=body)

    def delete(self, slug: str) -> Dict[str, Any]:
        """Delete a processor.

        **Irreversible, and the slug is burned forever** — it can never be reused, by you or
        anyone else. In-flight runs finish first; the code is wiped when they drain.
        """
        return self._request("DELETE", f"/processors/{slug}")

    def set_paused(self, slug: str, paused: bool) -> Dict[str, Any]:
        """Stop or restart traffic WITHOUT losing the slug. This is the switch, not ``delete``."""
        return self._request("PUT", f"/processors/{slug}/pause", json={"paused": paused})

    def set_listing(self, slug: str, listed: bool) -> Dict[str, Any]:
        """List or unlist publicly. Instant, no review step.

        Unlisting is NOT stopping: an unlisted processor keeps answering anyone holding the URL
        or an invoke token, earning nothing while still drawing compute from your balance. Use
        :meth:`set_paused`.
        """
        return self._request("PUT", f"/processors/{slug}/listing", json={"listed": listed})

    def set_secrets(self, slug: str, values: Dict[str, str]) -> Dict[str, Any]:
        """Set secret VALUES. Each name must already be declared in ``manifest.secrets``."""
        return self._request("PUT", f"/processors/{slug}/secrets", json={"values": values})

    def rotate_token(self, slug: str) -> Dict[str, Any]:
        """Mint a new invoke token. The old one stops working immediately."""
        return self._request("POST", f"/processors/{slug}/rotate-token")

    # -- observability ------------------------------------------------------

    def runs(self, slug: str) -> Dict[str, Any]:
        return self._request("GET", f"/processors/{slug}/runs")

    def get_run(self, slug: str, run_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/processors/{slug}/runs/{run_id}")

    def earnings(self, slug: str) -> Dict[str, Any]:
        """Sales (paid straight to your wallet, with the on-chain tx per row) and runtime spend."""
        return self._request("GET", f"/processors/{slug}/earnings")

    def kv(self, slug: str) -> Dict[str, Any]:
        """The processor's own key/value state. Read-only from out here, by design."""
        return self._request("GET", f"/processors/{slug}/kv")

    # -- webhooks -----------------------------------------------------------

    def get_webhook(self, slug: str) -> Dict[str, Any]:
        return self._request("GET", f"/processors/{slug}/webhook")

    def set_webhook(self, slug: str, url: str) -> Dict[str, Any]:
        """Register or replace the webhook.

        We immediately POST a signed verification to the URL: it must answer 2xx or the webhook
        stays registered-but-inactive and delivers nothing. The signing secret comes back
        EXACTLY ONCE.
        """
        return self._request("PUT", f"/processors/{slug}/webhook", json={"url": url})

    def delete_webhook(self, slug: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/processors/{slug}/webhook")

    def test_webhook(self, slug: str) -> Dict[str, Any]:
        return self._request("POST", f"/processors/{slug}/webhook/test")

    # -- invoking -----------------------------------------------------------

    def run(self, slug: str, input: Dict[str, Any], invoke_token: str) -> Any:
        """Run YOUR OWN processor with the invoke token from :meth:`deploy`.

        Not the API key: the run route deliberately does not read a key as an ownership claim,
        because it is the only route with both a money path and an anonymous buyer lane. You pay
        for the compute; nobody pays at call time.
        """
        return self._request(
            "POST",
            f"/processors/{slug}/run",
            json={"input": input},
            headers={"Authorization": f"Bearer {invoke_token}"},
            send_api_key=False,
        )

    def run_with_payment(
        self,
        slug: str,
        input: Dict[str, Any],
        payment_header: Optional[str] = None,
        accept_networks: Optional[Sequence[str]] = None,
    ) -> Any:
        """Run someone else's processor as a buyer, with an x402 payment header.

        Call once WITHOUT ``payment_header`` to get the 402 and its ``accepts`` array — one entry
        per chain that publisher takes. Match on ``network``, pay that entry, and retry with the
        header.

        Re-sending the SAME header returns the run that payment already bought and does NOT charge
        again. That is the recovery path for every failure mode, because **there are no refunds**:
        the money went straight to the publisher and the platform never held it.

        ``accept_networks`` opts in to Robinhood, which is withheld from the 402 unless asked for.
        The reference x402 client validates the WHOLE accepts array against a fixed chain list and
        throws on the first name it does not recognise, so advertising it unprompted would stop a
        conformant buyer paying on Solana or Base either.
        """
        headers: Dict[str, str] = {}
        if payment_header:
            headers["X-Payment"] = payment_header
        if accept_networks:
            headers["X-Accept-Networks"] = ",".join(accept_networks)
        return self._request(
            "POST",
            f"/processors/{slug}/run",
            json={"input": input},
            headers=headers or None,
            send_api_key=False,
        )
