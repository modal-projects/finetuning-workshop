import modal

app = modal.App()

train_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("unsloth==2025.8.4").uv_pip_install("wandb")

hf_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


with train_image.imports():
    # Import `unsloth` first -- it applies important patches.
    from unsloth import FastLanguageModel  # noqa
    from datasets import load_dataset
    import torch
    from trl import SFTConfig, SFTTrainer


@app.function(
    image=train_image,
    gpu="h200",
    volumes={"/root/.cache/huggingface": hf_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_dict({"WANDB_PROJECT": "finetuning-workshop"}),
    ],
)
def train(lora_r: int, lora_alpha: int, batch_size: int):
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=2048,
        dtype="bfloat16",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # Can make training 5x faster for short sequences.
        args=SFTConfig(
            disable_tqdm=True,
            per_device_train_batch_size=batch_size,  # Let's change this to higher.
            gradient_accumulation_steps=1,
            warmup_steps=5,
            # num_train_epochs=1,  # Set this for 1 full training run.
            max_steps=60,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",  # Use this for WandB etc
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    print(trainer_stats)


@app.local_entrypoint()
def main():
    for lora_r, lora_alpha, batch_size in [
        (8, 8, 2),
        (8, 8, 8),
        (16, 16, 2),
        (16, 16, 8),
        (32, 32, 2),
        (32, 32, 8),
    ]:
        train.spawn(lora_r, lora_alpha, batch_size)
