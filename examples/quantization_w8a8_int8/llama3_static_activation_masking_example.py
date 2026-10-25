"""
Static Activation Quantization Example with Token Masking

Quantizing activations to `int8` requires a calibration range. With a static
(default) strategy that range is fixed at calibration time from the observed
minimum and maximum, so the calibration data directly determines the scale of
every input tensor.

This example restricts the observed activations to the tokens the model
actually generates (assistant responses). Deep layers see very different
activations for prompt tokens than for generated tokens, so calibrating only on
the prompt can produce a range that is too wide for generation, and vice versa.

Both input and output observers honor the mask; only `input_activations` is
quantized here, which is the common case for `W8A8` models.

Usage:
    # Assistant-only activation ranges (default)
    python llama3_static_activation_masking_example.py

    # Baseline without masking, for comparison
    python llama3_static_activation_masking_example.py --no-use-loss-mask

Evaluate both checkpoints with lm-evaluation-harness, for example:
    lm_eval --model vllm \
        --model_args pretrained=<checkpoint>,max_model_len=4096 \
        --tasks gsm8k --num_fewshot 5 --batch_size auto
"""

import argparse
import os

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-loss-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Observe activations for assistant tokens only. Run with "
            "`--no-use-loss-mask` to produce the unmasked baseline checkpoint."
        ),
    )
    return parser.parse_args()


def preprocess(example, tokenizer):
    """
    Tokenize a conversation and mark the final assistant response with 1.

    The prompt is tokenized separately with `add_generation_prompt=True`, which
    is exactly the prefix the model conditions on at inference time. Every
    token of the response that follows is enabled in the mask.
    """
    messages = example["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)["input_ids"]

    # The prompt is not always an exact prefix of the rendered conversation
    # (for example when a tokenizer emits a different suffix on the final
    # turn). Such samples cannot be masked and are dropped by the filter below.
    aligned = len(prompt_ids) < len(full_ids) and list(
        full_ids[: len(prompt_ids)]
    ) == list(prompt_ids)

    loss_mask = [0] * len(full_ids)
    if aligned:
        loss_mask[len(prompt_ids) :] = [1] * (len(full_ids) - len(prompt_ids))

        # Left-truncate long conversations so the response is always retained.
        # The leading BOS token is kept, since the model always sees it first.
        if len(full_ids) > MAX_SEQUENCE_LENGTH:
            keep = MAX_SEQUENCE_LENGTH - 1
            full_ids = list(full_ids[:1]) + list(full_ids[-keep:])
            loss_mask = loss_mask[:1] + loss_mask[-keep:]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "loss_mask": loss_mask,
    }


def main():
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda example: preprocess(example, tokenizer),
        remove_columns=ds.column_names,
    )
    # Samples whose response could not be identified contribute no masked
    # tokens, so drop them. The filter is applied for both runs to keep the
    # comparison fair.
    ds = ds.filter(lambda example: any(example["loss_mask"]))

    # Quantize weights to int8 per channel and activations to int8 per tensor.
    # `dynamic=False` makes the activation ranges static, which is what the
    # calibration data is used for.
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="channel"
        ),
        input_activations=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="tensor", dynamic=False
        ),
    )
    recipe = QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    )

    print(f"Calibrating on {len(ds)} samples, use_loss_mask={args.use_loss_mask}")

    # With `use_loss_mask=True` the observers only see tokens where
    # `loss_mask == 1`.
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
    SAVE_DIR = f"{MODEL_ID.rstrip('/').split('/')[-1]}-W8A8-INT8-static-{suffix}"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
