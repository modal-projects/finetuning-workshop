### Simple demos getting Axolotl and Unsloth up and running on Modal

Axolotl vs Unsloth (2025-08-13 summary)
- `unsloth` has optimized training kernels for single GPUs.
- `axolotl` has better multi-GPU support, but is lower performance (especially when `lora_dropout` is non-zero).

Details:
- Fine-tune base model Meta-Llama-3.1-8B to be an Instruct model with Alpaca SFT dataset
