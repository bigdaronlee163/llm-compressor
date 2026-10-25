import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "nm-testing/tinysmokeqwen3"
NUM_SAMPLES = 4
MAX_SEQ_LENGTH = 64


def _recipe() -> GPTQModifier:
    return GPTQModifier(
        ignore=["lm_head"],
        config_groups={
            "group_0": QuantizationScheme(
                targets=["re:.*model.layers.2.self_attn.q_proj$"],
                weights=QuantizationArgs(num_bits=4, strategy="channel"),
            )
        },
    )


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


def _load_quantized_weight(checkpoint_dir):
    """Read the packed q_proj weight from a saved checkpoint."""
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        for key, value in load_file(shard).items():
            if key.endswith("layers.2.self_attn.q_proj.weight_packed"):
                return value

    raise AssertionError(f"Quantized q_proj weight not found in {checkpoint_dir}")


@pytest.mark.parametrize("pipeline", ["independent", "basic"])
def test_gptq_loss_mask_changes_quantized_weights(tmp_path, pipeline):
    device = 0 if torch.accelerator.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = _make_dataset(tokenizer)

    for use_loss_mask, name in ((True, "masked"), (False, "unmasked")):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=device)
        oneshot(
            model=model,
            dataset=dataset,
            recipe=_recipe(),
            num_calibration_samples=NUM_SAMPLES,
            max_seq_length=MAX_SEQ_LENGTH,
            use_loss_mask=use_loss_mask,
            pipeline=pipeline,
            output_dir=tmp_path / f"{name}-{pipeline}",
        )

    masked_weight = _load_quantized_weight(tmp_path / f"masked-{pipeline}")
    unmasked_weight = _load_quantized_weight(tmp_path / f"unmasked-{pipeline}")
    assert not torch.equal(masked_weight, unmasked_weight), (
        "Quantized weights are identical with and without loss masking. "
        "The loss mask was not applied during hessian accumulation."
    )
