# singularity-grid

Python SDK for the SGL Network confidential compute grid. Submit inference jobs to TEE-verified nodes, check grid capacity, and use the OpenAI-compatible chat completions endpoint -- all from a single package.

## Installation

```bash
pip install singularity-grid
```

To use the OpenAI compatibility helper:

```bash
pip install singularity-grid[openai]
```

## Quick start

### 1. OpenAI-compatible usage (simplest)

The SGL Network orchestrator exposes an OpenAI-compatible `/v1/chat/completions` endpoint. You can use the standard `openai` package with no wrapper:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://grid.x402compute.cc/v1",
    api_key="scg_your_api_key",
)

response = client.chat.completions.create(
    model="gemma-4-26b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

### 2. Helper function

If you prefer not to copy the URL, use the built-in helper:

```python
from singularity_grid import create_openai_client

client = create_openai_client(api_key="scg_your_api_key")

response = client.chat.completions.create(
    model="gemma-4-26b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

### 3. GridClient (full grid features)

For grid-specific features like job submission, TEE attestation, capacity checks, and pricing:

```python
from singularity_grid import GridClient

# Public endpoints -- no auth required
grid = GridClient()
capacity = grid.capacity()
models = grid.models()
pricing = grid.pricing()

print(f"Active nodes: {capacity.active_nodes}/{capacity.total_nodes}")
for m in models:
    print(f"  {m.id} -- {m.sgl_node_count} nodes")

# Authenticated endpoints
grid = GridClient(api_key="scg_your_api_key")

job = grid.submit_job(
    model="gemma-4-26b",
    input_payload={
        "messages": [{"role": "user", "content": "Analyze this data"}]
    },
    submitter_wallet="0xYourWallet",
    submitter_chain="base",
)

print(f"Job {job.job_id}: {job.status}")

# Retrieve result and attestation
result = grid.get_job(job.job_id)
attestation = grid.get_attestation(job.job_id)
print(f"TEE verified: {attestation.verified}")
```

## API reference

### GridClient

| Method | Auth | Description |
|---|---|---|
| `capacity()` | No | Grid-wide capacity summary |
| `models()` | No | Available models with pricing and TEE info |
| `pricing()` | No | Pricing table for all models |
| `submit_job(model, input_payload, ...)` | Yes | Submit a compute job |
| `get_job(job_id)` | Yes | Get job status and result |
| `get_attestation(job_id)` | Yes | Get TEE attestation proof |

### Exceptions

| Exception | When |
|---|---|
| `SGLError` | Base class for all SDK errors |
| `SGLAPIError` | Non-2xx response from the API |
| `SGLAuthError` | 401 or 403 response |
| `SGLNotFoundError` | 404 response |
| `SGLConnectionError` | Orchestrator unreachable or timeout |

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | Bearer token for authenticated endpoints |
| `base_url` | orchestrator URL | Override the orchestrator URL |
| `timeout` | `60.0` | Request timeout in seconds |

## License

MIT
