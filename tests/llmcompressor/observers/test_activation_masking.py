from types import SimpleNamespace

import pytest
import torch

import llmcompressor.modifiers.quantization.calibration as calibration
from llmcompressor.core import create_session
from llmcompressor.modifiers.quantization.calibration import (
    calibrate_input_hook,
    calibrate_output_hook,
)


class _Recorder:
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)


class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_observer = _Recorder()
        self.output_observer = _Recorder()
        self.quantization_scheme = SimpleNamespace(output_activations=None)


def _masked_state(session, loss_mask):
    session.state.loss_masks = [loss_mask]
    session.state.current_batch_idx = 0


@pytest.mark.unit
def test_input_hook_masks_observed_activations():
    module = _Module()
    x = torch.randn(2, 3, 4)
    loss_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)

    with create_session() as session:
        _masked_state(session, loss_mask)
        calibrate_input_hook(module, (x,))

    observed = module.input_observer.values[0]
    expected = x.reshape(-1, 4)[loss_mask.reshape(-1)]
    assert torch.equal(observed, expected)


@pytest.mark.unit
def test_input_hook_without_mask_observes_everything():
    module = _Module()
    x = torch.randn(2, 3, 4)

    with create_session():
        calibrate_input_hook(module, (x,))

    assert torch.equal(module.input_observer.values[0], x)


@pytest.mark.unit
def test_output_hook_masks_observer_but_not_returned_output(monkeypatch):
    module = _Module()
    x = torch.randn(2, 3, 4)
    loss_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
    monkeypatch.setattr(
        calibration, "forward_quantize", lambda **kwargs: kwargs["value"] + 1
    )

    with create_session() as session:
        _masked_state(session, loss_mask)
        returned = calibrate_output_hook(module, (), x)

    observed = module.output_observer.values[0]
    expected = x.reshape(-1, 4)[loss_mask.reshape(-1)]
    assert torch.equal(observed, expected)
    # the value which flows through the model must not be masked
    assert torch.equal(returned, x + 1)


@pytest.mark.unit
def test_mismatched_mask_keeps_full_activations():
    module = _Module()
    x = torch.randn(2, 3, 4)
    loss_mask = torch.ones(2, 2, dtype=torch.bool)

    with create_session() as session:
        _masked_state(session, loss_mask)
        calibrate_input_hook(module, (x,))

    assert torch.equal(module.input_observer.values[0], x)
