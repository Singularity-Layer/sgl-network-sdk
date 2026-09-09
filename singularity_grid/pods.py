"""Agent Pods — hosted agents you can create, drive and destroy over an API key.

A pod is a running agent on its own machine: it holds a conversation, keeps a workspace, can
be given scheduled tasks and connectors, and exposes an OpenAI-compatible endpoint so anything
that already speaks to OpenAI can speak to it instead.

Two things are worth knowing before you use this.

**Create is idempotent, and the key is not decoration.** Creating a pod provisions a machine
and charges for it, so a retry after a timeout must not do it twice. ``create_pod`` generates
an idempotency key when you do not pass one, and replaying the same key returns the original
response byte for byte rather than making a second pod. Pass your own key when your caller has
a natural id for the attempt (an order number, a job id) — that is what makes a retry after a
crash safe rather than merely likely to be safe.

**Delete is not instant.** Providers refuse to delete a machine that is still installing, so a
destroy can come back as ``destroying`` rather than ``destroyed``. ``delete_pod`` reports
which, and ``wait_for_destroyed`` holds until the machine is genuinely gone. Treating the first
as the second is how you keep paying for something you thought you deleted.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from .client import SGLAPIError, SGLAuthError, SGLConnectionError, SGLNotFoundError

DEFAULT_PODS_BASE_URL = "https://compute.x402layer.cc"
DEFAULT_TIMEOUT = 120.0


class PodsClient:
    """Client for the Agent Pods API.

    Args:
        api_key: mint one in the dashboard under Settings -> API Keys.
        base_url: override for testing against a local worker.
        timeout: seconds. Creating a pod is the slow call; it provisions a machine.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_PODS_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise SGLAuthError(401, "An api_key is required. Mint one under Settings -> API Keys.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "PodsClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ── transport ──────────────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        headers = {"X-API-Key": self.api_key, "Accept": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        try:
            resp = self._client.request(
                method, f"{self.base_url}/pods/v1{path}", json=json, headers=headers
            )
        except httpx.RequestError as exc:
            raise SGLConnectionError(f"Could not reach the pods API: {exc}") from exc

        try:
            parsed = resp.json() if resp.content else None
        except ValueError:
            parsed = None

        if resp.status_code >= 400:
            err = (parsed or {}).get("error", {}) if isinstance(parsed, dict) else {}
            message = err.get("message") or f"HTTP {resp.status_code}"
            # Echoed on every response, and the fastest way for us to find your request in our
            # logs — so it travels with the error rather than being dropped.
            request_id = resp.headers.get("x-request-id")
            if request_id:
                message = f"{message} (request id {request_id})"
            body = {"code": err.get("code"), "details": err.get("details"), "request_id": request_id}
            if resp.status_code in (401, 403):
                raise SGLAuthError(resp.status_code, message, body)
            if resp.status_code == 404:
                raise SGLNotFoundError(resp.status_code, message, body)
            raise SGLAPIError(resp.status_code, message, body)
        return parsed

    # ── pods ───────────────────────────────────────────────────────────────────────────

    def create_pod(
        self,
        tier: str = "starter",
        name: Optional[str] = None,
        external_ref: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        region: Optional[str] = None,
        term_hours: Optional[int] = None,
        capabilities: Optional[Dict[str, bool]] = None,
        ai: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a pod. Provisions a machine and charges for it.

        Returns as soon as the order exists, with ``status == "provisioning"``; the machine
        takes a few minutes to boot. Use :meth:`wait_for_online` if you need to wait.

        ``external_ref`` is YOUR id for this pod. It comes back on reads and filters
        :meth:`list_pods`, so you can find a pod again from your own database without storing
        ours.
        """
        body: Dict[str, Any] = {"tier": tier}
        for key, value in (
            ("name", name), ("external_ref", external_ref), ("model", model),
            ("system_prompt", system_prompt), ("region", region),
            ("term_hours", term_hours), ("capabilities", capabilities), ("ai", ai),
        ):
            if value is not None:
                body[key] = value
        return self._request(
            "POST", "/pods", body,
            {"Idempotency-Key": idempotency_key or f"sdk-{uuid.uuid4()}"},
        )["pod"]

    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")["pod"]

    def list_pods(
        self,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        external_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List pods, newest first. ``external_ref`` filters by your own id."""
        parts = []
        if limit:
            parts.append(f"limit={limit}")
        if cursor:
            parts.append(f"cursor={cursor}")
        if external_ref:
            parts.append(f"external_ref={external_ref}")
        query = f"?{'&'.join(parts)}" if parts else ""
        return self._request("GET", f"/pods{query}")

    def update_pod(self, pod_id: str, **patch: Any) -> Dict[str, Any]:
        """Change ``name``, ``model``, ``slug`` or ``auto_renew``."""
        return self._request("PATCH", f"/pods/{pod_id}", patch)["pod"]

    def delete_pod(self, pod_id: str) -> Dict[str, Any]:
        """Destroy a pod.

        ``destroyed`` means the machine is confirmed gone. ``destroying`` means the provider
        would not delete it yet — usually because it is still installing — and a sweep will
        retry; it may bill for a few more minutes. Do not treat the second as the first.
        """
        res = self._request("DELETE", f"/pods/{pod_id}")
        return {"status": res["pod"]["status"], "message": res.get("message", "")}

    def wait_for_online(
        self, pod_id: str, timeout: float = 900.0, interval: float = 10.0
    ) -> Dict[str, Any]:
        """Poll until the pod is online, or raise when it lands somewhere it cannot leave."""
        return self._wait(pod_id, {"online"}, {"destroyed", "destroying"}, timeout, interval)

    def wait_for_destroyed(
        self, pod_id: str, timeout: float = 900.0, interval: float = 10.0
    ) -> Dict[str, Any]:
        """Poll until the machine is genuinely gone, not merely accepted for teardown."""
        return self._wait(pod_id, {"destroyed"}, set(), timeout, interval)

    def _wait(
        self,
        pod_id: str,
        done: set,
        fatal: set,
        timeout: float,
        interval: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            pod = self.get_pod(pod_id)
            status = pod.get("status")
            if status in done:
                return pod
            if status in fatal:
                raise SGLAPIError(409, f"Pod {pod_id} is {status}; it will not come online.")
            if time.monotonic() >= deadline:
                raise SGLAPIError(504, f"Timed out waiting for pod {pod_id}; it is {status}.")
            time.sleep(interval)

    # ── talking to a pod ───────────────────────────────────────────────────────────────

    def endpoint_url(self, pod: Any) -> str:
        """The pod's OpenAI-compatible base URL.

        Point an OpenAI client at this with a key from :meth:`mint_pod_key` and use the model
        id ``agent-pod``. Anything that already speaks OpenAI works unchanged::

            from openai import OpenAI
            key = pods.mint_pod_key(pod["id"])["secret"]
            oa = OpenAI(base_url=pods.endpoint_url(pod), api_key=key)
            oa.chat.completions.create(model="agent-pod", messages=[...])
        """
        if isinstance(pod, str):
            return f"{self.base_url}/pods/{pod}/v1"
        endpoint = pod.get("endpoint") or {}
        return endpoint.get("base_url") or f"{self.base_url}/pods/{pod['id']}/v1"

    def mint_pod_key(
        self, pod_id: str, label: Optional[str] = None, daily_cap_usd: Optional[float] = None
    ) -> Dict[str, Any]:
        """Mint a key for the pod's own endpoint. The secret is returned ONCE."""
        body: Dict[str, Any] = {}
        if label is not None:
            body["label"] = label
        if daily_cap_usd is not None:
            body["daily_cap_usd"] = daily_cap_usd
        return self._request("POST", f"/pods/{pod_id}/keys", body)["key"]

    def list_pod_keys(self, pod_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/pods/{pod_id}/keys")["keys"]

    def revoke_pod_key(self, pod_id: str, key_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}/keys/{key_id}")

    # ── operating a pod ────────────────────────────────────────────────────────────────

    def usage(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}/usage")["usage"]

    def queue_action(self, pod_id: str, action: str) -> Dict[str, Any]:
        """Queue ``restart``, ``stop``, ``redeploy``, ``update``, ``diagnose`` or ``logs``.

        Applied on the pod's next check-in, usually within a minute, which is why this returns
        202 rather than pretending it already happened. ``diagnose`` and ``logs`` write their
        output back; read it from :meth:`get_actions`.
        """
        return self._request("POST", f"/pods/{pod_id}/actions", {"action": action})["action"]

    def get_actions(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}/actions")["actions"]

    def list_tasks(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}/tasks")

    def add_task(
        self,
        pod_id: str,
        name: str,
        kind: str,
        schedule: str,
        message: Optional[str] = None,
        session: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"name": name, "kind": kind, "schedule": schedule}
        if message is not None:
            body["message"] = message
        if session is not None:
            body["session"] = session
        return self._request("POST", f"/pods/{pod_id}/tasks", body)

    def remove_task(self, pod_id: str, job_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/pods/{pod_id}/tasks/{job_id}")

    def get_wallet(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}/wallet")["wallet"]

    def update_wallet(self, pod_id: str, **patch: Any) -> Dict[str, Any]:
        """Change a pod wallet's spend controls.

        Needs ``pods:wallet:write`` — a general key that manages pods must not be able to
        raise the cap on the money it can spend. The field is ``per_tx_cap_usd``, not
        ``spend_cap_usd``; the API lists the editable names if you get it wrong.
        """
        return self._request("PATCH", f"/pods/{pod_id}/wallet", patch)

    def add_connector(
        self,
        pod_id: str,
        name: str,
        url: str,
        transport: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Attach an MCP connector, giving the agent a new tool.

        Needs ``pods:control:write`` for the same reason: handing an agent new tools is not
        something a general-purpose key should do.
        """
        body: Dict[str, Any] = {"name": name, "url": url}
        if transport is not None:
            body["transport"] = transport
        if headers is not None:
            body["headers"] = headers
        return self._request("POST", f"/pods/{pod_id}/connectors", body)["connectors"]

    def remove_connector(self, pod_id: str, connector_id: str) -> List[Dict[str, Any]]:
        return self._request("DELETE", f"/pods/{pod_id}/connectors/{connector_id}")["connectors"]

    def update_backups(
        self, pod_id: str, enabled: bool, passphrase: Optional[str] = None
    ) -> Dict[str, Any]:
        """Turn automatic backups on or off. Setting a passphrase needs ``pods:wallet:write``."""
        body: Dict[str, Any] = {"enabled": enabled}
        if passphrase is not None:
            body["passphrase"] = passphrase
        return self._request("PATCH", f"/pods/{pod_id}/backups", body)["backups"]

    def telegram_join_code(self, pod_id: str) -> Dict[str, Any]:
        """Mint a Telegram join code for a group."""
        return self._request("POST", f"/pods/{pod_id}/channels/telegram/join-code")["join"]

    def telegram_join_status(self, pod_id: str) -> Dict[str, Any]:
        """Poll whether a group has claimed the code. Never echoes the code back."""
        return self._request("GET", f"/pods/{pod_id}/channels/telegram/join-code")["join"]

    def get_update_policy(self, pod_id: str) -> Dict[str, Any]:
        """Who decides WHEN this pod takes our updates.

        By default we do: the pod polls every six hours and applies whatever we answer,
        restarting its gateway for about forty seconds. Fine for a pod you run for yourself,
        wrong for pods you run for customers who did not choose that moment.
        """
        return self._request("GET", f"/pods/{pod_id}/updates")["updates"]

    def set_update_policy(self, pod_id: str, mode: str) -> Dict[str, Any]:
        """``auto`` or ``manual``.

        ``manual`` makes us keep answering with the version the pod already has, so it never
        updates itself; you apply it with ``queue_action(pod_id, "update")`` when it suits
        you. The hold expires after 30 days, because security fixes ride these bundles, and
        the response always names the date so it is never a surprise.
        """
        return self._request("PATCH", f"/pods/{pod_id}/updates", {"mode": mode})["updates"]

    def chat_ticket(self, pod_id: str) -> Dict[str, Any]:
        """A short-lived ticket for the pod's streaming chat socket.

        ``scope`` says what the socket will accept. Without ``pods:control:write`` it is
        ``chat``, conversation only. Most integrations want the OpenAI endpoint instead.
        """
        return self._request("POST", f"/pods/{pod_id}/chat-ticket")["chat"]

    def connect_telegram_group(
        self, pod_id: str, require_mention: Optional[bool] = None, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Attach the Telegram group that claimed the join code.

        Takes no chat id: it attaches only the group that claimed the code, because a claim
        proves somebody typed it inside that room. Existing groups are preserved.
        """
        body: Dict[str, Any] = {}
        if require_mention is not None:
            body["require_mention"] = require_mention
        if prompt is not None:
            body["prompt"] = prompt
        return self._request("POST", f"/pods/{pod_id}/channels/telegram/connect", body)["channel"]

    def approve_pairing(self, pod_id: str, code: str, channel: str = "telegram") -> Dict[str, Any]:
        """Approve someone to DM the agent.

        A stranger who finds the bot can DM it, and unlike a group nobody else sees that
        conversation, so the agent refuses unknown people and shows them a code. The code must
        come from the agent, so this approves a request somebody already made.
        """
        return self._request("POST", f"/pods/{pod_id}/channels/{channel}/pair", {"code": code})["pairing"]

    def wallet_send(self, pod_id: str, **body: Any) -> Dict[str, Any]:
        """Move funds out of the pod's wallet. Needs ``pods:wallet:write``.

        The spend cap is enforced server-side before the transfer and the pod holds no keys,
        so this cannot exceed the policy set by :meth:`update_wallet`.
        """
        return self._request("POST", f"/pods/{pod_id}/wallet/send", body)

    def wallet_pay_x402(self, pod_id: str, **body: Any) -> Dict[str, Any]:
        """Pay an x402 endpoint from the pod's wallet. Needs ``pods:wallet:write``."""
        return self._request("POST", f"/pods/{pod_id}/wallet/x402/pay", body)

    def list_connectors(self, pod_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/pods/{pod_id}/connectors")["connectors"]

    def get_backups(self, pod_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}/backups")["backups"]

    # ── events and webhooks ────────────────────────────────────────────────────────────

    def list_events(
        self,
        after: Optional[int] = None,
        limit: Optional[int] = None,
        type: Optional[str] = None,
        pod_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Page the event log.

        Paged by ``seq``, an integer that only goes up, NOT by timestamp — two events can
        share a millisecond, and a timestamp cursor would either skip one or repeat it
        forever. Store the ``next_after`` you get back and pass it as ``after`` next time.

        This is also the answer to a missed webhook: delivery is a cursor over these same
        rows, so anything a broken endpoint dropped is still here.
        """
        parts = []
        if after is not None:
            parts.append(f"after={after}")
        if limit:
            parts.append(f"limit={limit}")
        if type:
            parts.append(f"type={type}")
        if pod_id:
            parts.append(f"pod_id={pod_id}")
        query = f"?{'&'.join(parts)}" if parts else ""
        return self._request("GET", f"/events{query}")

    def create_webhook(
        self, url: str, event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Register a webhook. HTTPS only.

        The signing ``secret`` comes back ONCE. Store it — it is how you verify a delivery
        really came from us. Omit ``event_types`` to receive everything; an empty list is
        refused, because a subscription to nothing is a webhook that silently never fires.
        """
        body: Dict[str, Any] = {"url": url}
        if event_types is not None:
            body["event_types"] = event_types
        return self._request("POST", "/webhooks", body)["webhook"]

    def list_webhooks(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/webhooks")["webhooks"]

    def update_webhook(self, webhook_id: str, **patch: Any) -> Dict[str, Any]:
        return self._request("PATCH", f"/webhooks/{webhook_id}", patch)["webhook"]

    def delete_webhook(self, webhook_id: str) -> None:
        self._request("DELETE", f"/webhooks/{webhook_id}")


_SIG_RE = re.compile(r"^t=(\d+),v1=([0-9a-f]{64})$")


def verify_pod_webhook(
    raw_body: str,
    signature_header: str,
    secret: str,
    tolerance_sec: int = 300,
    now: Optional[float] = None,
) -> bool:
    """Verify a webhook delivery came from us and is recent.

    Pass the RAW request body, not a re-serialised object: re-encoding JSON changes bytes
    (key order, spacing) and the signature is over the bytes we sent.

    The timestamp is inside the signed string, so a captured delivery cannot be replayed later
    under a fresh one. ``tolerance_sec`` is what stops an old capture being accepted at all.
    """
    match = _SIG_RE.match((signature_header or "").strip())
    if not match:
        return False
    timestamp = int(match.group(1))
    current = int(now if now is not None else time.time())
    if abs(current - timestamp) > tolerance_sec:
        return False
    expected = hmac.new(
        secret.encode(), f"{timestamp}.{raw_body}".encode(), hashlib.sha256
    ).hexdigest()
    # compare_digest, not ==: a short-circuiting compare leaks how much of a forged signature
    # was right, which is enough to reconstruct one byte at a time.
    return hmac.compare_digest(match.group(2), expected)
