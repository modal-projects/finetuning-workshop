### Simple demo getting Unsloth up and running on Modal

Goals:
- Demonstrate capability for training LoRA adapter on multiple GPUs
- Use supervised fine-tuning to elicit a particular response format

Why Unsloth?
- `axolotl` is higher compatibility, but lower performance.
- `unsloth` is a more realistic "start-from-scratch" demo - I haven't used it before!

Details:
- Fine-tune Meta-Llama-3.1-8B with a custom chat template
