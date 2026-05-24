"""Use the SGL Network with the standard OpenAI Python SDK.

The SGL Network orchestrator exposes an OpenAI-compatible /v1/chat/completions
endpoint, so you can use the openai package directly with no code changes
beyond swapping the base URL and API key.
"""

# --- Option 1: raw OpenAI SDK (no singularity-grid import needed) ----------

from openai import OpenAI

client = OpenAI(
    base_url="https://sgl-network-orchestrator.ivaavimusicproductions.workers.dev/v1",
    api_key="scg_your_api_key_here",
)

response = client.chat.completions.create(
    model="gemma-4-26b",
    messages=[{"role": "user", "content": "Hello from the SGL Network!"}],
)
print(response.choices[0].message.content)


# --- Option 2: use the helper from singularity-grid -----------------------

from singularity_grid import create_openai_client

client2 = create_openai_client(api_key="scg_your_api_key_here")

response2 = client2.chat.completions.create(
    model="gemma-4-26b",
    messages=[{"role": "user", "content": "What TEE type processed this?"}],
)
print(response2.choices[0].message.content)

# List available models
models = client2.models.list()
for m in models.data:
    print(f"  {m.id} (owned by {m.owned_by})")
