import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
)
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "nm-testing/tinysmokellama-3.2"
NUM_SAMPLES = 4
MAX_SEQ_LENGTH = 64


def _make_dataset(tokenizer) -> Dataset:
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs."
    )
    rows = []
    for _ in range(NUM_SAMPLES):
        input_ids = tokenizer(
            text, max_length=MAX_SEQ_LENGTH, truncation=True, return_tensors="pt"
        )["input_ids"][0]
        loss_mask = torch.ones_like(input_ids)
        loss_mask[: input_ids.numel() // 2] = 0
        rows.append(
            {
                "input_ids": input_ids.tolist(),
                "attention_mask": [1] * input_ids.numel(),
                "loss_mask": loss_mask.tolist(),
            }
        )

    return Dataset.from_list(rows)


def _load_tensor(checkpoint_dir, suffix):
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        for key, value in load_file(shard).items():
            if key.endswith(suffix):
                return value

    raise AssertionError(f"{suffix} not found in {checkpoint_dir}")


def _quantize(tmp_path, tokenizer, recipe_fn, name, use_loss_mask):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    dataset = _make_dataset(tokenizer)
    output_dir = tmp_path / name
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe_fn(),
        num_calibration_samples=NUM_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH,
        use_loss_mask=use_loss_mask,
        output_dir=output_dir,
    )
    return output_dir


def _imatrix_recipe():
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=True,
            strategy="group",
            group_size=32,
            observer="imatrix_mse",
        ),
    )
    return QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])


def _static_activation_recipe():
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=True, strategy="group", group_size=32
        ),
        input_activations=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="tensor", dynamic=False
        ),
    )
    return QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])


def test_imatrix_observer_masking_changes_weight_scales(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    masked_dir = _quantize(tmp_path, tokenizer, _imatrix_recipe, "masked", True)
    unmasked_dir = _quantize(tmp_path, tokenizer, _imatrix_recipe, "unmasked", False)

    masked_scale = _load_tensor(masked_dir, "q_proj.weight_scale")
    unmasked_scale = _load_tensor(unmasked_dir, "q_proj.weight_scale")
    assert not torch.equal(masked_scale, unmasked_scale), (
        "Weight scales are identical with and without loss masking. "
        "The loss mask was not applied to the imatrix importance."
    )


def test_activation_observer_masking_changes_input_scales(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    masked_dir = _quantize(
        tmp_path, tokenizer, _static_activation_recipe, "masked", True
    )
    unmasked_dir = _quantize(
        tmp_path, tokenizer, _static_activation_recipe, "unmasked", False
    )

    masked_scale = _load_tensor(masked_dir, "q_proj.input_scale")
    unmasked_scale = _load_tensor(unmasked_dir, "q_proj.input_scale")
    assert not torch.equal(masked_scale, unmasked_scale), (
        "Activation scales are identical with and without loss masking. "
        "The loss mask was not applied to the activation observers."
    )
