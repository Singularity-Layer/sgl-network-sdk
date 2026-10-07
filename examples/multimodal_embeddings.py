"""Create one ordered text + image EmbeddingGemma 2 vector."""

import os

from singularity_grid import (
    EMBEDDINGGEMMA2_MODEL,
    GridClient,
    image_part,
    media_from_file,
    multimodal_item,
    text_part,
)

grid = GridClient(api_key=os.environ["SGL_API_KEY"])

product = multimodal_item(
    text_part("A compact red travel backpack"),
    image_part(media_from_file("./backpack.png")),
)
response = grid.embeddings(
    EMBEDDINGGEMMA2_MODEL,
    [product],
    dimensions=256,
    input_type="document",
)

vector = response["data"][0]["embedding"]
print(f"dimensions={len(vector)} usage={response['usage']['breakdown']}")
