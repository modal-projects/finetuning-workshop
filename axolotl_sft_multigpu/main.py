"""
We can launch a training run on all the configs.

modal run -m axolotl_sft_multigpu.main
"""

from pathlib import Path
from datetime import datetime

import modal

axolotl_image = (
    modal.Image.from_registry("axolotlai/axolotl:main-20250812-py3.11-cu126-2.7.1")
    .pip_install("huggingface_hub", "hf-transfer", "wandb", "fastapi", "pydantic")
    .env(
        dict(
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
    .add_local_dir(Path(__file__).parent / "configs", "/root/configs")
)

app = modal.App(
    "axolotl-sft-multigpu",
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

hf_cache_volume = modal.Volume.from_name("axolotl-huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("axolotl-sft-multigpu-checkpoints", create_if_missing=True)


@app.function(
    image=axolotl_image,
    gpu="H100:4",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=4 * 60 * 60,  # 4 hours
)
def train(config: str, output_dir: str):
    import subprocess

    subprocess.run(
        f"axolotl train /root/configs/{config} --output-dir /checkpoints/{output_dir}", shell=True, check=True
    )


@app.local_entrypoint()
def main():
    datestring = datetime.now().strftime("%Y%m%d-%H%M%S")
    handles = [
        train.spawn("config-fft.yml", output_dir=datestring + "-out"),
        train.spawn("config-lora.yml", output_dir=datestring + "-out-lora"),
        train.spawn("config-lora-faster.yml", output_dir=datestring + "-out-lora-faster"),
        train.spawn("config-lora-faster-oom.yml", output_dir=datestring + "-out-lora-faster-oom"),
    ]
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            print("Training failed: ", e)
