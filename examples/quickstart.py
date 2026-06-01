"""Quickstart: submit a job to the SGL Network and retrieve its result."""

from singularity_grid import GridClient

# Authenticated client for job submission
grid = GridClient(api_key="scg_your_api_key_here")

# Submit an inference job
job = grid.submit_job(
    model="gemma-4-26b",
    input_payload={
        "messages": [{"role": "user", "content": "Explain zero-knowledge proofs in one paragraph."}]
    },
    submitter_wallet="0xYourWalletAddress",
    submitter_chain="base",
)

print(f"Job submitted: {job.job_id} (status: {job.status})")
print(f"Estimated cost: ${job.estimated_cost_usd}")

# Poll for result (in production, use a callback or webhook)
result = grid.get_job(job.job_id)
print(f"Job status: {result.status}")

if result.status == "completed":
    print(f"Result: {result.result}")

    # Verify TEE attestation
    attestation = grid.get_attestation(job.job_id)
    print(f"TEE type: {attestation.tee_type}")
    print(f"Verified: {attestation.verified}")
