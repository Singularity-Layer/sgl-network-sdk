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

### 4. Multimodal embeddings

EmbeddingGemma 2 returns one normalized vector per input item. Each item can contain ordered text,
image, audio, and video parts. Media stays inline in the confidential request and must include a
SHA-256 digest. The file helper reads only supported, bounded local files and creates the canonical
base64 envelope for you.

```python
from singularity_grid import (
    EMBEDDINGGEMMA2_MODEL,
    GridClient,
    image_part,
    media_from_file,
    multimodal_item,
    text_part,
)

grid = GridClient(api_key="scg_your_api_key")
item = multimodal_item(
    text_part("A compact red travel backpack"),
    image_part(media_from_file("./backpack.png")),
)
response = grid.embeddings(
    EMBEDDINGGEMMA2_MODEL,
    [item],
    dimensions=256,
    input_type="document",
    encoding_format="float",
)
print(response["data"][0]["embedding"])
print(response["usage"]["breakdown"])
```

Existing text calls are unchanged: both `"one string"` and `["one", "two"]` remain valid.
EmbeddingGemma 2 supports dimensions `768`, `512`, `256`, and `128`; `input_type` can be
`"query"`, `"document"`, or `"unspecified"`. See
[`examples/multimodal_embeddings.py`](examples/multimodal_embeddings.py) for a runnable example.

Supported MIME types are exported as `EMBEDDINGGEMMA2_MIME_TYPES`: JPEG, PNG, WebP, WAV, FLAC,
MP3, and MP4. Platform aliases inferred for `.wav` and `.flac` files are normalized to the exact
Grid values `audio/wav` and `audio/flac`.

The public constants `EMBEDDINGGEMMA2_LIMITS` describe the request limits. Key limits are 16 batch
items, 16 ordered parts per item, 8 images per item, one audio and one video part per item, 30
seconds of audio, 32 seconds of video, and 20 MiB of decoded media per request. Local validation
raises `EmbeddingInputError` with a stable `.code`. API failures raise `SGLAPIError`; its `.code`
contains the server's stable error code when present.

### 5. Confidential transcription (private v1)

`transcribe_pcm` accepts one raw, headerless PCM utterance. Bytes must already be mono, 16 kHz,
signed 16-bit little-endian PCM. The SDK rejects empty, partial-sample, and over-60-second inputs
before network access, reserves an eligible node using metadata only, and seals the audio directly
to that node's verified X25519 key. It verifies the node signature before decrypting the result.

```python
from singularity_grid import GridClient

grid = GridClient(api_key="x402c_your_api_key")
result = grid.transcribe_pcm_file("utterance.pcm", language="en")  # or "auto"

print(result["text"])
print(result["segments"])
```

Use `transcribe_pcm(pcm_bytes, ...)` when audio is already in memory. This private v1 route is
client-sealed JSON; it is not multipart, streaming, or OpenAI wire-compatible. It does not accept
WAV/MP3 containers or remote URLs. The route can remain unavailable while Grid transcription is
dark. A submit is never retried automatically because a timeout can leave paid settlement in an
ambiguous state; retain the logical `request_id` when reconciling a request.

Billable duration is exact `sample_count / 16000`, including fractional seconds. The pinned
rate is $0.0001 per audio second, with a $0.0001 minimum charge and rounding up to whole
micro-USDC. The SDK recomputes both quote and final charge; `job_id` must equal the original
request UUID. After a timeout or `outcome_unknown`, reconcile that UUID before any explicit retry.
The SDK does not sign x402 wallet payments; use API-key credits for this client.

Request IDs must be canonical lowercase UUIDv4 values; the SDK generates one when omitted.
Silent audio can return an empty transcript with no segments. Segment text has a combined
64 KiB UTF-8 limit, and timestamp starts/ends must be nondecreasing, allowing at most 50 ms overlap.

The package exports the exact model commit and SHA-256, protocol, audio format, and limits.
Local input failures raise `TranscriptionInputError`. Substituted reservations, bad node key
bindings/signatures, unsupported envelopes, and result binding mismatches raise
`TranscriptionResponseError`. Neither exception includes audio or transcript data.

### 6. System One / Laya

Laya is served as a typed-decision model, not as chat completions.

```python
from singularity_grid import GridClient

grid = GridClient(api_key="x402c_your_api_key")

systemone_models = grid.systemone_models()
print([model["id"] for model in systemone_models])

decision = grid.system_one(
    model="convaiinnovations/laya",
    state={"ticket": "Enterprise customer cannot access billing exports"},
    questions={
        "route": {
            "type": "choice",
            "instructions": "Choose the best team.",
            "criteria": {
                "billing": "Billing, invoice, refund, or account credit issue.",
                "support": "Product defect or technical troubleshooting.",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Score urgency from 0 to 1.",
        },
    },
)
print(decision["answers"])
```

## API reference

### GridClient

| Method | Auth | Description |
|---|---|---|
| `capacity()` | No | Grid-wide capacity summary |
| `models()` | No | Available models with pricing and TEE info |
| `v1_models(type=None)` | No | OpenAI-compatible model list; pass `type="systemone"` for Laya |
| `systemone_models()` | No | List Laya/System One typed-decision models |
| `pricing()` | No | Pricing table for all models |
| `embeddings(model, input, ...)` | Yes | Typed text or ordered multimodal embeddings |
| `embed(model, input, ...)` | Yes | Embedding vectors only, ordered like the input |
| `transcribe_pcm(pcm, ...)` | Yes | Validate, client-seal, and transcribe raw 16 kHz mono s16 PCM |
| `transcribe_pcm_file(path, ...)` | Yes | Bounded raw-PCM file helper |
| `system_one(state, questions, ...)` | Yes | Call `/v1/systemone` for typed decisions |
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
| `EmbeddingInputError` | Local media, input, dimension, or limit validation failure; inspect `.code` |
| `TranscriptionInputError` | Local PCM, language, request ID, or limit validation failure; inspect `.code` |
| `TranscriptionResponseError` | Reservation key binding, signed envelope, or result binding failed |

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` | Bearer token for authenticated endpoints |
| `base_url` | orchestrator URL | Override the orchestrator URL |
| `timeout` | `60.0` | Request timeout in seconds |
| `transcription_timeout` | `120.0` | STT-only reserve/submit timeout in seconds |
| `transcription_canary_token` | `None` | Private token sent only to STT reserve/submit routes |

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
