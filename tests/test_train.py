"""client.train tests — httpx.MockTransport only, no network.

Route paths and field names in the fake mirror SGLCompute_Backend/src/handlers/
training.ts; if the server changes, change the fake to match the server, never
the other way around.
"""

import json

import httpx
import pytest

from singularity_grid import GridClient
from singularity_grid.client import SGLAPIError, SGLAuthError

COMPUTE = "https://compute.test"


def make_client(handler):
    c = GridClient(api_key="sk_test", compute_url=COMPUTE)
    c.train._client = httpx.Client(
        base_url=COMPUTE,
        headers={"Accept": "application/json", "X-API-Key": "sk_test"},
        transport=httpx.MockTransport(handler),
    )
    return c


def test_train_requires_api_key():
    c = GridClient(compute_url=COMPUTE)  # no api_key
    with pytest.raises(SGLAuthError):
        c.train


def test_catalog_sends_api_key_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers.get("X-API-Key")
        return httpx.Response(200, json={"models": [], "gpus": [], "quantize": [], "constraints": {}, "image": "t"})

    cat = make_client(handler).train.catalog()
    assert seen["path"] == "/training/catalog"
    assert seen["key"] == "sk_test"
    assert cat["image"] == "t"


def test_create_dataset_presign_put_approve_flow(tmp_path, monkeypatch):
    calls = []
    put_seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/training/datasets/presign":
            body = json.loads(request.content)
            assert body["bytes"] == 11
            return httpx.Response(201, json={
                "dataset_id": "d" * 8 + "-1111-1111-1111-111111111111",
                "upload_url": "https://r2.test/bucket/key?sig=1",
                "expires_in": 900, "max_bytes": 33554432, "format": "jsonl",
            })
        if request.url.path.endswith("/approve"):
            return httpx.Response(200, json={
                "dataset_id": "d" * 8, "approved_at": "2026-08-25T00:00:00Z",
                "row_count": 2, "total_rows": 2, "dropped": 0, "reasons": {},
            })
        raise AssertionError(f"unexpected {request.url}")

    def fake_put(url, content=None, headers=None, **kwargs):
        put_seen["url"] = url
        put_seen["content"] = content
        put_seen["headers"] = headers
        return httpx.Response(200)

    monkeypatch.setattr("singularity_grid.train.httpx.put", fake_put)

    f = tmp_path / "data.jsonl"
    f.write_bytes(b"12345678901")
    out = make_client(handler).train.create_dataset(f)
    assert out.row_count == 2
    assert calls == [("POST", "/training/datasets/presign"), ("POST", "/training/datasets/dddddddd-1111-1111-1111-111111111111/approve")]
    assert put_seen["url"] == "https://r2.test/bucket/key?sig=1"
    assert put_seen["headers"] is None or "X-API-Key" not in put_seen["headers"]  # presigned — no auth header leak


def test_synthesize_posts_exact_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/training/datasets/synth"
        body = json.loads(request.content)
        assert set(body) == {"description", "seed_examples", "system_prompt", "target_rows"}
        return httpx.Response(202, json={"dataset_id": "d", "status": "pending", "charged": 1.0, "target_rows": 200, "note": "n"})

    seeds = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}] * 5
    out = make_client(handler).train.synthesize("teach it", seeds, system_prompt="sys", target_rows=200)
    assert out["status"] == "pending"


def test_create_run_maps_gpu_and_hours():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/training/runs"
        body = json.loads(request.content)
        assert body == {
            "dataset_id": "d1", "base_model": "qwen2.5-0.5b-instruct",
            "gpu_plan_id": "runpod-rtx-3090-24g", "budget_hours": 2,
        }
        return httpx.Response(201, json={"run_id": "r1", "status": "provisioning", "charged": 1.2, "billed_hours": 2, "gpu": "RTX 3090", "base_model": body["base_model"], "note": "n"})

    out = make_client(handler).train.create_run("d1", "qwen2.5-0.5b-instruct", gpu="runpod-rtx-3090-24g", hours=2)
    assert out.run_id == "r1"


def test_create_run_requires_gpu_and_hours():
    c = make_client(lambda r: httpx.Response(500))
    with pytest.raises(TypeError):
        c.train.create_run("d1", "qwen2.5-0.5b-instruct")  # no gpu/hours → no defaults


def test_insufficient_credits_raises_payment_required():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": "Insufficient credits", "required": 1.2})

    with pytest.raises(SGLAPIError) as exc_info:
        make_client(handler).train.create_run("d1", "m", gpu="g", hours=1)
    assert exc_info.value.status_code == 402


def test_wait_polls_until_terminal(monkeypatch):
    states = iter([
        {"run": {"id": "r1", "status": "running", "outcome": "running"}},
        {"run": {"id": "r1", "status": "destroyed", "outcome": "completed", "exit_code": 0}},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(states))

    monkeypatch.setattr("singularity_grid.train.time.sleep", lambda s: None)
    final = make_client(handler).train.wait("r1", poll_s=0)
    assert final.outcome == "completed"


def test_cancel_and_extend_paths():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"run_id": "r1", "status": "cancelled", "refunded": 0, "note": "Prepaid blocks are not refunded on cancel."})
        assert json.loads(request.content) == {"add_hours": 3}
        return httpx.Response(200, json={"run_id": "r1", "added_hours": 3, "charged": 1.8, "new_deadline_at": None, "reference": "train-extend:r1:1"})

    t = make_client(handler).train
    assert t.cancel("r1")["refunded"] == 0
    assert t.extend("r1", 3)["added_hours"] == 3
    assert seen == ["/training/runs/r1/cancel", "/training/runs/r1/extend"]


def test_download_writes_artifacts(tmp_path, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/deploy")
        assert json.loads(request.content) == {"target": "download"}
        return httpx.Response(200, json={
            "run_id": "r1", "target": "download",
            "urls": {"adapter": "https://r2.test/a", "report": "https://r2.test/r"},
            "expires_in": 3600, "gguf_available": False,
        })

    class FakeStreamResponse:
        def __init__(self, url):
            self.status_code = 200
            self._content = b"bytes-" + url.encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self):
            yield self._content

    def fake_stream(method, url, **kwargs):
        assert method == "GET"
        return FakeStreamResponse(url)

    monkeypatch.setattr("singularity_grid.train.httpx.stream", fake_stream)

    written = make_client(handler).train.download("r1", tmp_path)
    assert sorted(p.name for p in written) == ["adapter.tar.gz", "report.json"]
    assert (tmp_path / "adapter.tar.gz").read_bytes() == b"bytes-https://r2.test/a"


def test_download_skips_unknown_keys(tmp_path, monkeypatch):
    """Verify that path-traversal hostile keys like ../../evil are skipped (not written)."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/deploy")
        assert json.loads(request.content) == {"target": "download"}
        return httpx.Response(200, json={
            "run_id": "r1", "target": "download",
            "urls": {
                "adapter": "https://r2.test/a",
                "../../evil": "https://r2.test/evil",  # Hostile key: path traversal attempt
                "/etc/passwd": "https://r2.test/passwd",  # Hostile key: absolute path
                "report": "https://r2.test/r",
            },
            "expires_in": 3600, "gguf_available": False,
        })

    class FakeStreamResponse:
        def __init__(self, url):
            self.status_code = 200
            self._content = b"bytes-" + url.encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self):
            yield self._content

    def fake_stream(method, url, **kwargs):
        assert method == "GET"
        return FakeStreamResponse(url)

    monkeypatch.setattr("singularity_grid.train.httpx.stream", fake_stream)

    written = make_client(handler).train.download("r1", tmp_path)
    # Only whitelisted keys (adapter, report) should be written; hostile keys are skipped.
    assert sorted(p.name for p in written) == ["adapter.tar.gz", "report.json"]
    # Verify no traversal happened: ../../evil should not exist anywhere.
    assert not (tmp_path / "../../evil").resolve().exists()  # safety check: no escape
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "etc").exists()


def test_no_fake_hf_surface():
    t = make_client(lambda r: httpx.Response(500)).train
    assert not hasattr(t, "push_to_hub")
    assert not hasattr(t, "deploy_endpoint")
