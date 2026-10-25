"""
GPTQ Quantization Example with Token Masking

This example demonstrates GPTQ quantization with token masking to focus the
hessian accumulation on assistant responses only. The loss_mask feature makes
GPTQ ignore user prompts and special tokens during calibration, so the
quantized weights better preserve the tokens the model actually generates.

This is particularly useful for instruction-tuned and reasoning models, where
the prompt can dominate the calibration sequence while generation-time quality
is what matters.

Usage:
    # Assistant-only calibration (default)
    python llama3_gptq_masking_example.py

    # Baseline without masking, for comparison
    python llama3_gptq_masking_example.py --no-use-loss-mask

    # Smaller model for a faster run
    MODEL_ID=meta-llama/Llama-3.2-1B-Instruct python llama3_gptq_masking_example.py

Evaluate both checkpoints with lm-evaluation-harness, for example:
    lm_eval --model vllm \
        --model_args pretrained=<checkpoint>,max_model_len=4096 \
        --tasks gsm8k --num_fewshot 5 --batch_size auto
"""

import argparse
import os
from functools import partial

from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Select model and load it.
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

# Select calibration dataset.
# ultrachat_200k is a multi-turn conversation dataset with user/assistant messages.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-loss-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict GPTQ calibration to assistant response tokens. Run with "
            "`--no-use-loss-mask` to produce the unmasked baseline checkpoint."
        ),
    )
    return parser.parse_args()


def mask_from_llama3_headers(input_ids, tokenizer):
    """
    Build a loss mask by locating Llama-3 assistant headers.

    Used as a fallback when the tokenizer's chat template does not support
    `return_assistant_tokens_mask`.

    Llama-3 chat format:
        <|start_header_id|>assistant<|end_header_id|>\n\n[content]<|eot_id|>
    """
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")

    mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids) - 2:
        if (
            input_ids[i] == start_header_id
            and input_ids[i + 1] == assistant_id
            and input_ids[i + 2] == end_header_id
        ):
            i += 3  # Skip header tokens
            while i < len(input_ids) and input_ids[i] != eot_id:
                mask[i] = 1
                i += 1
        else:
            i += 1

    return mask


def preprocess(example, tokenizer):
    """
    Tokenize a conversation and mark assistant response tokens with 1.

    The mask is 1 for tokens the assistant generates and 0 for everything else
    (user turns, headers, and special tokens).
    """
    try:
        tokenized = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
        )
        loss_mask = tokenized.get("assistant_masks")
    except ValueError:
        # Chat template does not declare generation spans
        tokenized = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            return_dict=True,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
        )
        loss_mask = None

    input_ids = tokenized["input_ids"]
    if loss_mask is None or not any(loss_mask):
        loss_mask = mask_from_llama3_headers(input_ids, tokenizer)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "loss_mask": loss_mask,
    }


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        partial(preprocess, tokenizer=tokenizer), remove_columns=ds.column_names
    )

    # Drop samples where the assistant portion was truncated away, since they
    # would not contribute any tokens to the masked calibration. The filter is
    # applied for both runs so that the comparison uses identical samples.
    ds = ds.filter(lambda example: any(example["loss_mask"]))

    # Configure the quantization algorithm to run.
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

    if args.use_loss_mask:
        print(f"Calibrating with token masking on {len(ds)} samples")
    else:
        print(f"Calibrating without token masking on {len(ds)} samples")

    # Apply algorithms. With `use_loss_mask=True`, GPTQ accumulates the hessian
    # only over tokens where `loss_mask == 1`.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        use_loss_mask=args.use_loss_mask,
    )

    # Confirm generations of the quantized model look sane.
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    dispatch_model(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save to disk compressed.
    suffix = "masked" if args.use_loss_mask else "unmasked"
    SAVE_DIR = f"{MODEL_ID.rstrip('/').split('/')[-1]}-W4A16-G128-{suffix}"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
