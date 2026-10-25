import pytest
import torch
from torch.utils.data import DataLoader

import llmcompressor.pipelines.basic.pipeline as basic_pipeline
from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.pipelines.basic import BasicPipeline


class _StrictModel(torch.nn.Module):
    """Model whose forward rejects any keyword other than the ones below."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None):
        return input_ids


class _NoopLifecycleCallbacks:
    @staticmethod
    def calibration_start(**kwargs):
        pass

    @staticmethod
    def sequential_epoch_end(*args, **kwargs):
        pass

    @staticmethod
    def calibration_end(**kwargs):
        pass


def _batch(with_loss_mask: bool) -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    if with_loss_mask:
        batch["loss_mask"] = torch.tensor([[0, 1, 1]])
    return batch


@pytest.mark.unit
def test_basic_pipeline_does_not_forward_loss_mask(monkeypatch):
    monkeypatch.setattr(basic_pipeline, "LifecycleCallbacks", _NoopLifecycleCallbacks)
    dataloader = DataLoader([_batch(with_loss_mask=True)], batch_size=None)
    dataset_args = DatasetArguments(use_loss_mask=True)

    with create_session() as session:
        BasicPipeline()(_StrictModel(), dataloader, dataset_args)

        assert len(session.state.loss_masks) == 1
        assert torch.equal(session.state.loss_masks[0], torch.tensor([[0, 1, 1]]))


@pytest.mark.unit
def test_basic_pipeline_pops_loss_mask_when_masking_disabled(monkeypatch):
    monkeypatch.setattr(basic_pipeline, "LifecycleCallbacks", _NoopLifecycleCallbacks)
    dataloader = DataLoader([_batch(with_loss_mask=True)], batch_size=None)
    dataset_args = DatasetArguments(use_loss_mask=False)

    with create_session() as session:
        # a leftover loss_mask field must not be forwarded to the model
        BasicPipeline()(_StrictModel(), dataloader, dataset_args)

        assert session.state.loss_masks is None


@pytest.mark.unit
def test_basic_pipeline_requires_loss_mask_when_enabled(monkeypatch):
    monkeypatch.setattr(basic_pipeline, "LifecycleCallbacks", _NoopLifecycleCallbacks)
    dataloader = DataLoader([_batch(with_loss_mask=False)], batch_size=None)
    dataset_args = DatasetArguments(use_loss_mask=True)

    with create_session():
        with pytest.raises(ValueError, match="loss_mask"):
            BasicPipeline()(_StrictModel(), dataloader, dataset_args)
