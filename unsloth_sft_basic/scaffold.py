import modal

app = modal.App()

# train_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("unsloth==2025.8.4")

# hf_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


# @app.function(
#     image=train_image,
#     gpu="h200",
#     volumes={"/root/.cache/huggingface": hf_vol},
# )
# def train():
#     pass
