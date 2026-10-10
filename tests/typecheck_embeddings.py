"""Static compatibility assertions for the overloaded embeddings surface."""

from typing import Any, Dict, Optional

from singularity_grid import EmbeddingResponse, GridClient, TranscriptionResponse

client = GridClient(base_url="https://grid.test")

# Existing callers commonly keep model, input, and input_type broad. This assignment pins the
# pre-0.11 source contract: broad inputs still produce a dictionary type.
dynamic_model: str = "nomic-embed-text-v1.5"
dynamic_input: Any = ["one", "two"]
dynamic_input_type: Optional[str] = "document"
dynamic_encoding_format: Optional[str] = "float"
legacy_response: Dict[str, Any] = client.embeddings(
    dynamic_model,
    dynamic_input,
    dimensions=64,
    input_type=dynamic_input_type,
    encoding_format=dynamic_encoding_format,
)

# A literal EmbeddingGemma 2 model selects the narrower multimodal request and response types.
typed_response: EmbeddingResponse = client.embeddings(
    "embeddinggemma-2",
    ["one", "two"],
    dimensions=128,
    input_type="unspecified",
    encoding_format="float",
)

transcription_response: TranscriptionResponse = client.transcribe_pcm(
    b"\x00\x00", language="en"
)

transcription_file_response: TranscriptionResponse = client.transcribe_pcm_file("utterance.pcm", language="en")
