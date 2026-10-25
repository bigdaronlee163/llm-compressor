import pytest
import torch
from datasets import Dataset

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import (
    _PaddingCollatorWithLossMask,
    format_calibration_data,
)


def _dataset(with_loss_mask: bool) -> Dataset:
    data = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }
    if with_loss_mask:
        data["loss_mask"] = [[0, 1, 1]]
    return Dataset.from_dict(data)


def _fake_padding_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    return {
        "input_ids": torch.tensor(
            [
                feature["input_ids"] + [0] * (max_len - len(feature["input_ids"]))
                for feature in features
            ]
        )
    }


@pytest.mark.unit
def test_format_calibration_data_requires_loss_mask_column():
    args = DatasetArguments(use_loss_mask=True, num_calibration_samples=1)

    with pytest.raises(ValueError, match="loss_mask"):
        format_calibration_data(args, _dataset(with_loss_mask=False), processor=None)


@pytest.mark.unit
def test_format_calibration_data_accepts_loss_mask_column():
    args = DatasetArguments(use_loss_mask=True, num_calibration_samples=1)

    dataloader = format_calibration_data(
        args, _dataset(with_loss_mask=True), processor=None
    )
    batch = next(iter(dataloader))

    assert "loss_mask" in batch
    assert batch["loss_mask"].tolist() == [[0, 1, 1]]


@pytest.mark.unit
@pytest.mark.parametrize(
    "padding_side,expected",
    [
        ("right", [[0, 1, 1], [1, 0, 0]]),
        ("left", [[0, 1, 1], [0, 1, 0]]),
    ],
)
def test_padding_collator_pads_loss_mask(padding_side, expected):
    collator = _PaddingCollatorWithLossMask(
        _fake_padding_collator, padding_side=padding_side
    )
    features = [
        {"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
        {"input_ids": [4, 5], "loss_mask": [1, 0]},
    ]

    batch = collator(features)

    assert batch["loss_mask"].tolist() == expected
    # input features must not be mutated
    assert "loss_mask" in features[0]
