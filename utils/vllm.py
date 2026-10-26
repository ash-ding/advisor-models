"""Start a local vLLM OpenAI-compatible server for the advisor model.

The upstream repo imports this module from `utils/eval_utils.py` but does not
ship it. vLLM lives in the training venv (SkyRL/skyrl-train/.venv), not in the
root venv the eval scripts run in, so the server is launched with that
interpreter (override with `VLLM_PYTHON`).
"""

import os
import subprocess
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_PYTHON = os.environ.get(
    "VLLM_PYTHON", str(REPO_ROOT / "SkyRL" / "skyrl-train" / ".venv" / "bin" / "python")
)
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))


def start_vllm_server(
    model_to_serve_name: str,
    served_model_name: str = "advisor_model",
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    port: int = VLLM_PORT,
    startup_timeout: float = 1800.0,
) -> subprocess.Popen:
    """Launch vLLM and block until its /health endpoint answers 200."""
    cmd = [
        VLLM_PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_to_serve_name,
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print("Launching vLLM:", " ".join(cmd))
    process = subprocess.Popen(cmd)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=5).ok:
                return process
        except requests.RequestException:
            pass
        time.sleep(5)
    process.terminate()
    raise TimeoutError(f"vLLM did not become healthy within {startup_timeout}s")
