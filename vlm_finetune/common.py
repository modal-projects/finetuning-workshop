import modal

from pathlib import Path

CONFIG_FILE_PATH = Path("/config.yml")

axolotl_image = (
    modal.Image.from_registry("axolotlai/axolotl:0.12.1")
    .pip_install(
        "huggingface_hub",
        "hf-transfer",
        "wandb",
        "fastapi",
        "pydantic",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
    .add_local_file(Path(__file__).parent / "config.yml", CONFIG_FILE_PATH.as_posix())
)

checkpoints_volume = modal.Volume.from_name("axolotl-vlm-finetune-checkpoints", create_if_missing=True)

data_volume = modal.Volume.from_name("axolotl-vlm-finetune-data", create_if_missing=True)