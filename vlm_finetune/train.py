import modal
from pathlib import Path

from .common import axolotl_image, checkpoints_volume, data_volume, CONFIG_FILE_PATH

app = modal.App("axolotl-vlm-finetune")

CKPT_VOLUME_DIR = Path("/checkpoints")
DATA_VOLUME_DIR = Path("/data")

LORA_OUTPUT_DIR = CKPT_VOLUME_DIR / "lora-out"
MERGED_OUTPUT_DIR = CKPT_VOLUME_DIR / "merged-out"


@app.function(
    image=axolotl_image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        CKPT_VOLUME_DIR.as_posix(): checkpoints_volume,
        DATA_VOLUME_DIR.as_posix(): data_volume
    },
    timeout=4 * 60 * 60, # 4 hours
)
def train():
    import subprocess

    subprocess.run(
        [
            "axolotl",
            "train",
            CONFIG_FILE_PATH.as_posix(),
            "--output-dir",
            LORA_OUTPUT_DIR.as_posix(),
        ],
        check=True,
    )

    subprocess.run(
        [
            "axolotl",
            "merge-lora",
            CONFIG_FILE_PATH.as_posix(),
            f"--lora-model-dir={LORA_OUTPUT_DIR.as_posix()}",
            f"--output-dir={MERGED_OUTPUT_DIR.as_posix()}",
        ],
        check=True,
    )