"""Anthropic Messages API on the Singularity Layer Grid.

    SGL_API_KEY=x402c_... python examples/anthropic_messages.py

Model ids are Grid model ids. Claude Code users: set ANTHROPIC_BASE_URL=https://grid.x402compute.cc
and ANTHROPIC_API_KEY to your compute key, and pick a large-context model (Claude Code's own
prompt is about 13k tokens; today qwen-2.5-32b at 32k fits).
"""

import os

from singularity_grid import GridClient

MODEL = os.environ.get("SGL_MODEL", "qwen-2.5-32b")
QUESTION = [{"role": "user", "content": "Name three uses of a trusted execution environment."}]

with GridClient(api_key=os.environ["SGL_API_KEY"]) as grid:
    print("estimated input tokens:", grid.count_tokens(MODEL, QUESTION)["input_tokens"])

    msg = grid.messages(MODEL, QUESTION, max_tokens=256, system="Be brief.")
    for block in msg["content"]:
        if block["type"] == "text":
            print(block["text"])
    print("stop:", msg["stop_reason"], "usage:", msg["usage"])

    # Tools: the model answers with a tool_use block you run yourself.
    weather_tool = {
        "name": "get_weather",
        "description": "Current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
    with_tool = grid.messages(
        MODEL, [{"role": "user", "content": "What is the weather in Paris?"}], max_tokens=256, tools=[weather_tool]
    )
    print(with_tool["stop_reason"], with_tool["content"])

    # Streaming
    for ev in grid.messages_stream(MODEL, [{"role": "user", "content": "Count from 1 to 5."}], max_tokens=256):
        if ev["type"] == "content_block_delta" and ev["delta"].get("type") == "text_delta":
            print(ev["delta"]["text"], end="", flush=True)
    print()
