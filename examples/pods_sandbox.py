"""Agent Pods sandbox: run a command, edit a file, serve a port, clean up.

    SGL_API_KEY=x402c_... POD_ID=<an online pod you own> python examples/pods_sandbox.py

The key needs the pods:control:write scope. Nothing here creates or pays for a pod.
"""

import os

from singularity_grid import PodsClient

pod_id = os.environ["POD_ID"]

with PodsClient(api_key=os.environ["SGL_API_KEY"]) as pods:
    run = pods.exec(pod_id, "uname -a && whoami", timeout_seconds=30)
    print("exit", run["exit_code"], run["stdout"])

    pods.write_file(pod_id, "index.html", "<h1>Hello from my pod</h1>")
    print(pods.read_file(pod_id, "index.html")["content"])

    # Serve the workspace on 8080 and give it a public URL (shown once).
    pods.exec(pod_id, "nohup python3 -m http.server 8080 >/dev/null 2>&1 &", timeout_seconds=10)
    port = pods.open_port(pod_id, 8080, "public")["port"]
    print("open", port["url"])

    print(pods.list_ports(pod_id))
    pods.close_port(pod_id, 8080)
