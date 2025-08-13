import modal
import subprocess
import re


from .common import axolotl_image, data_volume, CONFIG_FILE_PATH
from pathlib import Path
import json

DATA_VOLUME_DIR = Path("/data")

app = modal.App("axolotl-vlm-prep-dataset")

@app.function(
    image=axolotl_image,
    gpu="H100",
    volumes={DATA_VOLUME_DIR.as_posix(): data_volume},
)
def prep_dataset():
    USER_TAG = "<|user|>"
    ASSISTANT_TAG = "<|assistant|>"
    END_TAG = "<|end|>"

    def to_messages(row):
        ctx = row.get("context", "")
        # keep only the user's turn (between <|user|> ... <|assistant|>)
        if USER_TAG in ctx:
            ctx = ctx.split(USER_TAG, 1)[1]
        if ASSISTANT_TAG in ctx:
            ctx = ctx.split(ASSISTANT_TAG, 1)[0]
        ctx = ctx.replace(END_TAG, "").strip()

        images = row.get("images", []) or []

        # Build multimodal content parts for the chat template
        content_parts = []

        def extract_image_url(img):
            # Accept either string URLs/paths or dicts with common keys
            if isinstance(img, str):
                return img
            else:
                return (
                    img.get("path")
                    or img.get("url")
                    or img.get("image_url")
                )

        def to_image_part(img):
            url = extract_image_url(img)
            return {"type": "image", "image_url": url} if url else None

        for img in images:
            part = to_image_part(img)
            if part.get("image_url"):
                content_parts.append(part)

        # Remove any legacy inline image placeholders from text context
        ctx_no_tokens = re.sub(r"<\|image_\d+\|>", "", ctx).strip()

        if ctx_no_tokens:
            content_parts.append({"type": "text", "text": ctx_no_tokens})

        assistant_text = (row.get("response") or "").strip()

        # Normalize top-level images column to list[str]
        normalized_images = [u for u in (extract_image_url(img) for img in images) if u]

        return {
            "messages": [
                {"role": "user", "content": content_parts},
                {
                    "role": "assistant",
                    "content": ([{"type": "text", "text": assistant_text}] if assistant_text else []),
                },
            ],
            "images": normalized_images,
        }

    src_paths = [
        DATA_VOLUME_DIR / "train.jsonl",
        DATA_VOLUME_DIR / "eval.jsonl",
        DATA_VOLUME_DIR / "test.jsonl",
    ]

    def convert_one(src: Path) -> None:
        if not src.exists():
            return
        dst = src.with_suffix("")
        dst = dst.with_name(dst.name + ".messages.jsonl")

        with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                src_obj = json.loads(line)
                dst_obj = to_messages(src_obj)
                fout.write(json.dumps(dst_obj, ensure_ascii=False) + "\n")

    for p in src_paths:
        convert_one(p)


    subprocess.run(
        [
            "axolotl",
            "preprocess",
            CONFIG_FILE_PATH.as_posix(),
        ],
        check=True,
    )
    