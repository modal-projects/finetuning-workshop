import modal

from .common import data_volume

app = modal.App("vllm-qwen-test")

@app.function(
        image=modal.Image.debian_slim().pip_install("requests"),
        volumes={"/data": data_volume}
)
def test():
    import os
    import json
    import re
    import requests
    import modal

    def get_server_base_url() -> str:
        url = os.environ.get("VLLM_SERVER_URL")
        if url:
            return url.rstrip("/")
        try:
            fn = modal.Function.lookup("vllm-qwen-inference", "serve")
            web_url = getattr(fn, "web_url", None)
            if web_url:
                return web_url.rstrip("/")
        except Exception:
            pass
        # Fallback for local/dev if the web URL cannot be resolved inside Modal.
        return "http://127.0.0.1:8000"

    def normalize_label(text: str) -> str:
        if text is None:
            return ""
        t = text.strip().strip('"').strip("'. ")
        # Extract first token and decide yes/no
        first = re.split(r"\s+|[.,;:!?]", t)[0].lower() if t else ""
        if first.startswith("y"):
            return "yes"
        if first.startswith("n"):
            return "no"
        # As a last resort, search anywhere in the string
        if re.search(r"\byes\b", t, flags=re.IGNORECASE):
            return "yes"
        if re.search(r"\bno\b", t, flags=re.IGNORECASE):
            return "no"
        return t.lower()

    def extract_expected_answer(messages):
        expected_text_parts = []
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = m.get("content")
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text" and "text" in part:
                            expected_text_parts.append(part["text"])\
                                
                elif isinstance(content, str):
                    expected_text_parts.append(content)
                break
        expected_text = " ".join(expected_text_parts).strip() if expected_text_parts else ""
        return normalize_label(expected_text)

    def strip_assistant_messages(messages):
        return [m for m in messages if m.get("role") != "assistant"]

    base_url = get_server_base_url()
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    data_path = "/data/test.messages.jsonl"
    if not os.path.exists(data_path):
        print(f"Missing test data at {data_path}")
        return

    total = 0
    correct = 0
    errors = 0

    with open(data_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                errors += 1
                print(f"[{line_idx}] JSON decode error: {e}")
                continue

            messages = row.get("messages", [])
            expected = extract_expected_answer(messages)
            input_messages = strip_assistant_messages(messages)

            payload = {
                "model": model_name,
                "messages": input_messages,
                "max_tokens": 8,
                "temperature": 0,
            }

            try:
                resp = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                errors += 1
                print(f"[{line_idx}] Request error: {e}")
                continue

            try:
                content = data["choices"][0]["message"]["content"]
                predicted = normalize_label(content)
            except Exception as e:
                errors += 1
                print(f"[{line_idx}] Parse error: {e}; raw: {str(data)[:500]}")
                continue

            is_correct = predicted == expected and predicted in {"yes", "no"}
            total += 1
            correct += 1 if is_correct else 0

            print(
                f"[{line_idx}] expected={expected!r} predicted={predicted!r} correct={is_correct}"
            )

    if total == 0 and errors == 0:
        print("No test rows processed.")
    else:
        print(
            f"Done. total={total} correct={correct} accuracy={(correct/total*100.0 if total else 0):.2f}% errors={errors}"
        )