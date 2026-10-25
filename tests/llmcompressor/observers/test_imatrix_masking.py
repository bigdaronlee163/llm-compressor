import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs

from llmcompressor.core import create_session
from llmcompressor.observers.base import Observer


def _make_observer(module: torch.nn.Module) -> Observer:
    args = QuantizationArgs(
        num_bits=4,
        symmetric=True,
        strategy="channel",
        observer="imatrix_mse",
    )
    observer = Observer.load_from_registry("imatrix_mse", base_name="weight", args=args)
    observer.attach(module)
    return observer


@pytest.mark.unit
def test_imatrix_masks_importance_accumulation():
    torch.manual_seed(0)
    module = torch.nn.Linear(8, 4)
    observer = _make_observer(module)

    x = torch.randn(2, 3, 8)
    loss_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)

    with create_session() as session:
        session.state.loss_masks = [loss_mask]
        session.state.current_batch_idx = 0
        module(x)

    kept = x.reshape(-1, 8)[loss_mask.reshape(-1)]
    assert observer._imatrix_count.item() == kept.shape[0]
    assert torch.allclose(observer._imatrix_sum, kept.pow(2).sum(dim=0))


@pytest.mark.unit
def test_imatrix_without_mask_accumulates_all_tokens():
    module = torch.nn.Linear(8, 4)
    observer = _make_observer(module)

    x = torch.ones(2, 3, 8)

    with create_session():
        module(x)

    assert observer._imatrix_count.item() == 6
    assert torch.allclose(observer._imatrix_sum, torch.full((8,), 6.0))


@pytest.mark.unit
def test_imatrix_mismatched_mask_raises():
    module = torch.nn.Linear(8, 4)
    _make_observer(module)

    x = torch.randn(2, 3, 8)
    loss_mask = torch.ones(2, 2, dtype=torch.bool)

    with create_session() as session:
        session.state.loss_masks = [loss_mask]
        session.state.current_batch_idx = 0

        with pytest.raises(ValueError, match="Loss mask has"):
            module(x)
