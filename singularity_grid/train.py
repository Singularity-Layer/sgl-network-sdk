"""TrainAPI — managed fine-tuning on the compute worker (client.train.*).

Mirrors SGLCompute_Backend/src/handlers/training.ts, the REAL v1 surface: deploy
is presigned downloads only, so there is NO push_to_hub / deploy_endpoint here.
Auth: the GridClient api_key sent as X-API-Key (compute API key from
cloud.x402compute.cc → Settings → API Keys).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from .models import (
    TrainArtifacts,
    TrainDatasetApproval,
    TrainDatasetInfo,
    TrainRunCreated,
    TrainRunInfo,
)

DEFAULT_COMPUTE_URL = "https://compute.x402layer.cc"

_LIVE_STATUSES = {"draft", "dataset_ready", "paid", "provisioning", "running", "destroying"}

# Allowlist of artifact keys (defense against path traversal from compromised worker).
_ARTIFACT_KEYS = {"adapter", "gguf", "report"}
_DEFAULT_NAMES = {"adapter": "adapter.tar.gz", "report": "report.json", "gguf": "model.gguf"}


class TrainAPI:
    def __init__(self, api_key: Optional[str], compute_url: str = DEFAULT_COMPUTE_URL, timeout: float = 60.0) -> None:
        from .client import SGLAuthError
        if not api_key:
            raise SGLAuthError(401, "Managed training requires GridClient(api_key=...)")
        self._client = httpx.Client(
            base_url=compute_url.rstrip("/"),
            headers={"Accept": "application/json", "X-API-Key": api_key},
            timeout=timeout,
        )

    def _request(self, method: str, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from .client import SGLConnectionError, _raise_for_status
        try:
            response = self._client.request(method, path, json=json)
        except httpx.ConnectError as exc:
            raise SGLConnectionError(f"Could not connect to the compute worker: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise SGLConnectionError(f"Request to {path} timed out: {exc}") from exc
        _raise_for_status(response)
        return response.json()  # type: ignore[no-any-return]

    # ─── catalog ─────────────────────────────────────────────────────────────

    def catalog(self) -> Dict[str, Any]:
        """Base models, GPU plans (train_price_hourly), quantize options, constraints."""
        return self._request("GET", "/training/catalog")

    # ─── datasets ────────────────────────────────────────────────────────────

    def create_dataset(self, file: Union[str, Path, bytes]) -> TrainDatasetApproval:
        """Upload JSONL → presigned R2 PUT → server-side approve. Raises SGLAPIError
        (422) with the server's validation payload on schema failure."""
        data = file if isinstance(file, bytes) else Path(file).read_bytes()
        pres = self._request("POST", "/training/datasets/presign", {"bytes": len(data)})
        put = httpx.put(pres["upload_url"], content=data, timeout=600)  # presigned — never send the API key
        if put.status_code >= 400:
            from .client import SGLAPIError
            raise SGLAPIError(put.status_code, "dataset upload failed")
        return self.approve(pres["dataset_id"])

    def synthesize(
        self,
        description: str,
        seeds: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        target_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Queue prompt→dataset synthesis (async, 202). Poll dataset(id).synth_status
        until 'complete', then call approve(id) — approval is a separate, explicit step."""
        body: Dict[str, Any] = {"description": description, "seed_examples": seeds}
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if target_rows is not None:
            body["target_rows"] = target_rows
        return self._request("POST", "/training/datasets/synth", body)

    def dataset(self, dataset_id: str) -> TrainDatasetInfo:
        data = self._request("GET", f"/training/datasets/{dataset_id}")
        return TrainDatasetInfo.model_validate(data.get("dataset", {}))

    def approve(self, dataset_id: str) -> TrainDatasetApproval:
        data = self._request("POST", f"/training/datasets/{dataset_id}/approve")
        return TrainDatasetApproval.model_validate(data)

    # ─── runs ────────────────────────────────────────────────────────────────

    def create_run(
        self,
        dataset_id: str,
        base_model: str,
        gpu: str,
        hours: float,
        recipe: Optional[Dict[str, Any]] = None,
        quantize: Optional[str] = None,
    ) -> TrainRunCreated:
        """Prepaid block: charge then provision. `gpu` (catalog gpu id) and `hours`
        are required with no defaults. Early completion / manual cancel do NOT refund."""
        body: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "base_model": base_model,
            "gpu_plan_id": gpu,
            "budget_hours": hours,
        }
        if recipe is not None:
            body["recipe"] = recipe
        if quantize is not None:
            body["quantize"] = quantize
        return TrainRunCreated.model_validate(self._request("POST", "/training/runs", body))

    def runs(self) -> List[TrainRunInfo]:
        data = self._request("GET", "/training/runs")
        return [TrainRunInfo.model_validate(r) for r in data.get("runs", [])]

    def run(self, run_id: str) -> TrainRunInfo:
        data = self._request("GET", f"/training/runs/{run_id}")
        return TrainRunInfo.model_validate(data.get("run", {}))

    def wait(self, run_id: str, poll_s: float = 15.0, timeout_s: Optional[float] = 3600) -> TrainRunInfo:
        """Poll run(run_id) until terminal; returns the final run.

        Args:
            run_id: The run to monitor.
            poll_s: Polling interval in seconds.
            timeout_s: Maximum wait time in seconds; defaults to 1 hour. Pass None for unbounded."""
        started = time.monotonic()
        while True:
            r = self.run(run_id)
            if r.status == "destroyed" or r.status not in _LIVE_STATUSES:
                return r
            if r.status == "destroying" and r.outcome in ("completed", "failed", "cancelled"):
                return r
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                raise TimeoutError(f"run {run_id} still {r.status} after {timeout_s}s")
            time.sleep(poll_s)

    def cancel(self, run_id: str) -> Dict[str, Any]:
        """Cancel a live run. NO refund — the block was bought, not metered."""
        return self._request("POST", f"/training/runs/{run_id}/cancel")

    def extend(self, run_id: str, hours: float) -> Dict[str, Any]:
        """New upfront debit that moves the deadline (409 near the artifact window)."""
        return self._request("POST", f"/training/runs/{run_id}/extend", {"add_hours": hours})

    # ─── artifacts (deploy v1 = download only) ───────────────────────────────

    def artifacts(self, run_id: str) -> TrainArtifacts:
        data = self._request("POST", f"/training/runs/{run_id}/deploy", {"target": "download"})
        return TrainArtifacts.model_validate(data)

    def download(self, run_id: str, dest: Union[str, Path]) -> List[Path]:
        """Fetch all artifacts into `dest`; returns written paths."""
        art = self.artifacts(run_id)
        dest_dir = Path(dest)
        dest_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for key, url in art.urls.items():
            # Allowlist keys (defense against path traversal from compromised worker).
            if key not in _ARTIFACT_KEYS:
                continue
            # Guard caller-supplied names (if any) by extracting basename only.
            filename = _DEFAULT_NAMES[key]
            out = dest_dir / filename
            with httpx.stream("GET", url, timeout=600) as resp:  # presigned — never send the API key
                if resp.status_code >= 400:
                    from .client import SGLAPIError
                    raise SGLAPIError(resp.status_code, f"artifact download failed for {key}")
                with open(out, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            written.append(out)
        return written

    def close(self) -> None:
        self._client.close()
