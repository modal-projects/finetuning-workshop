import modal

from .common import checkpoints_volume

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})  # faster model transfers
)

MODEL_PATH = "/checkpoints/merged-out/merged"  # Local path to Qwen/Qwen2.5-VL-7B-Instruct weights
VLLM_PORT = 8000


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("vllm-qwen-inference")

@app.function(
    image=vllm_image,
    gpu="H100",
    scaledown_window=240, # 4 minutes
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/checkpoints": checkpoints_volume,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_PATH,
        "--served-model-name",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--no-enforce-eager",
        # "--tensor-parallel-size", str(N_GPU)
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)