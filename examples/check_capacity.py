"""Check grid capacity, available models, and pricing (no auth required)."""

from singularity_grid import GridClient

grid = GridClient()

# Grid capacity
capacity = grid.capacity()
print(f"Nodes: {capacity.active_nodes}/{capacity.total_nodes} active")
for tee in capacity.by_tee_type:
    print(f"  {tee.tee_type}: {tee.active_nodes}/{tee.total_nodes}")

# Available models
print("\nModels:")
models = grid.models()
for m in models:
    tees = ", ".join(m.sgl_tee_types) if m.sgl_tee_types else "none"
    print(f"  {m.id} -- {m.sgl_node_count} nodes, TEE: {tees}")

# Pricing
print("\nPricing:")
pricing = grid.pricing()
for p in pricing:
    print(
        f"  {p.model}: "
        f"${p.price_per_1k_input_tokens_usd}/1k input, "
        f"${p.price_per_1k_output_tokens_usd}/1k output"
    )
