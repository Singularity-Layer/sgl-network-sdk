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

### 4. System One (typed decisions)

System One models such as Laya return typed answers to your questions about a JSON state. They
do not generate text. There are three answer types:

- `choice` picks one option from `criteria`.
- `score` gives a numeric score against a list of criteria.
- `noul` gives a single string, number or boolean.

You pay for input tokens only. You must supply `api_key`, because the Python client uses credits
and does not sign x402 payments.

```python
from singularity_grid import GridClient

grid = GridClient(api_key="scg_your_api_key")

grid.systemone.models()   # GET /v1/models?type=systemone

res = grid.systemone.create(
    model="convaiinnovations/laya",   # the alias "laya" also works
    state={"ticket": "I was charged twice this month"},
    questions={
        "route": {"type": "choice", "instructions": "Which queue?",
                  "criteria": {"billing": "Money problems", "tech": "Bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["The customer is blocked"]},
        "lang": {"type": "noul", "instructions": "ISO language code of the ticket"},
    },
    tier="standard",          # optional: "standard" | "confidential"
)

res.answers["route"].choice      # "billing"
res.answers["urgency"].score     # 0.7
res.answers["lang"].value        # "en"
res.usage.cost_usd
```

Questions can also be `SystemOneChoiceQuestion`, `SystemOneScoreQuestion` or
`SystemOneNoulQuestion` models. `create()` also accepts `task`, `lang` and `user`.

- A 402 error of type `payment_required` means that you did not supply `api_key`.
- Other 402 errors keep the message from the server, for example `insufficient_credits` and
  `pod_cap_reached`.
- `create()` raises `SGLNotFoundError` and `models()` returns `[]` while System One is off on
  the orchestrator.
- The client does not seal the state. It goes to the orchestrator over TLS.

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
| `systemone.models()` | No | System One models (`GET /v1/models?type=systemone`) |
| `systemone.create(model, state, questions, ...)` | Yes | Typed System One answers (`POST /v1/systemone`) |

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

## Processors

Processors are served by `processors.x402compute.cc`, not the grid, so they have their own client.

```python
from singularity_grid import ProcessorsClient

p = ProcessorsClient(api_key=os.environ["SGL_API_KEY"])

p.catalogue()                          # public, no credential
p.list()                               # yours          (processors:read)
p.deploy(manifest, code)               # returns the invoke token ONCE
p.update("my-processor", code=code)
p.set_listing("my-processor", True)
p.run("my-processor", {"name": "world"}, invoke_token)
```

> **`processors:write` is full control** of processors owned by that key's wallet — delete and
> secrets included, the same as a Cloudflare API token. Mint `processors:read` if you want a
> credential that cannot change anything. Note that compute keys do not expire, there is no audit
> log, and **delete is permanent**: the code is wiped and the slug is burned forever.

`run()` takes the **invoke token** from `deploy()`, not the API key — the run route is the only one
with both a money path and an anonymous buyer lane, so it does not read a key as an ownership
claim. Buyers use `run_with_payment()` with an x402 header instead.

### Upgrading from 0.8.x

The six processor methods on `GridClient` (`deploy_processor`, `invoke_processor`,
`list_processors`, `get_processor`, `delete_processor`, `get_processor_logs`) are **removed**, along
with the `Processor*` models. They pointed at `/grid/processors`, which has never existed — every
call returned 404 — and their models described an older design that was never shipped. Use
`ProcessorsClient`. Nothing else changed: chat, embeddings, jobs, models, capacity, pricing,
reserve and the vault are untouched.
