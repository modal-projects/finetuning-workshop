from pathlib import Path

import modal

CONFIG_FILE_PATH = Path("/config.yml")

axolotl_image = (
    modal.Image.from_registry("axolotlai/axolotl:main-20250813-py3.11-cu126-2.6.0")
    .pip_install(
        "huggingface_hub",
        "hf-transfer",
        "wandb",
        "fastapi",
        "pydantic",
    )
    .env(
        dict(
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
    .add_local_file(Path(__file__).parent / "config.yml", CONFIG_FILE_PATH.as_posix())
)

app = modal.App(
    "axolotl-vlm-finetune",
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

hf_cache_volume = modal.Volume.from_name("axolotl-huggingface-cache", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("axolotl-vlm-finetune-checkpoints", create_if_missing=True)


@app.function(
    image=axolotl_image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=4 * 60 * 60,  # 4 hours
)
def train():
    import subprocess

    subprocess.run(f"axolotl train {CONFIG_FILE_PATH.as_posix()}", shell=True)
