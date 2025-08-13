"""
Let's improve our training workflow a little bit.

- Speed: let's add FSDP (https://huggingface.co/docs/accelerate/en/usage_guides/fsdp).
- Observability: let's add wandb.
- Training: one full epoch.
"""

import modal

app = modal.App()

train_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("unsloth==2025.8.4")

hf_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="h200:4",
    volumes={"/root/.cache/huggingface": hf_vol},
    timeout=3600 * 4,
)
def train():
    import subprocess

    subprocess.run(
        "torchrun --nproc_per_node 4 -m step2.launch_training",
        shell=True,
    )
