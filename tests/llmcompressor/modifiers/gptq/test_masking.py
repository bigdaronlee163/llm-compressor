import pytest
import torch

from llmcompressor.modifiers.gptq.helpers import (
    accumulate_hessian,
    make_empty_hessian,
    prepare_batch,
)
from llmcompressor.modifiers.utils.masking import (
    get_active_loss_mask,
    select_masked_tokens,
)


def _expected_hessian(tokens: torch.Tensor) -> torch.Tensor:
    tokens = tokens.to(dtype=torch.float32)
    return 2 * tokens.t() @ tokens


@pytest.mark.unit
def test_accumulate_hessian_with_loss_mask():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 3, bias=False)
    inp = torch.randn(2, 5, 4)
    loss_mask = torch.tensor([[1, 0, 1, 1, 0], [0, 0, 1, 1, 1]]).to(torch.bool)

    hessian, num_samples = accumulate_hessian(
        inp,
        module,
        make_empty_hessian(module),
        torch.zeros(()),
        loss_mask=loss_mask,
    )

    kept_tokens = inp.reshape(-1, 4)[loss_mask.reshape(-1)]
    assert torch.allclose(hessian, _expected_hessian(kept_tokens))
    # sample counting is unchanged by masking: one sample per batch item
    assert num_samples.item() == inp.shape[0]


@pytest.mark.unit
def test_accumulate_hessian_without_loss_mask():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 3, bias=False)
    inp = torch.randn(2, 5, 4)

    hessian, num_samples = accumulate_hessian(
        inp,
        module,
        make_empty_hessian(module),
        torch.zeros(()),
    )

    assert torch.allclose(hessian, _expected_hessian(inp.reshape(-1, 4)))
    assert num_samples.item() == inp.shape[0]


@pytest.mark.unit
def test_accumulate_hessian_mismatched_loss_mask():
    module = torch.nn.Linear(4, 3, bias=False)
    inp = torch.randn(2, 5, 4)
    loss_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="Loss mask has"):
        accumulate_hessian(
            inp,
            module,
            make_empty_hessian(module),
            torch.zeros(()),
            loss_mask=loss_mask,
        )


@pytest.mark.unit
def test_accumulate_hessian_conv2d_with_loss_mask():
    module = torch.nn.Conv2d(2, 3, kernel_size=3)
    inp = torch.randn(1, 2, 8, 8)
    loss_mask = torch.ones(1, 64, dtype=torch.bool)

    with pytest.raises(ValueError, match="not supported for Conv2d"):
        accumulate_hessian(
            inp,
            module,
            make_empty_hessian(module),
            torch.zeros(()),
            loss_mask=loss_mask,
        )


@pytest.mark.unit
def test_accumulate_hessian_fully_masked_batch():
    module = torch.nn.Linear(4, 3, bias=False)
    inp = torch.randn(1, 5, 4)
    loss_mask = torch.zeros(1, 5, dtype=torch.bool)

    hessian, _ = accumulate_hessian(
        inp,
        module,
        make_empty_hessian(module),
        torch.zeros(()),
        loss_mask=loss_mask,
    )

    assert torch.count_nonzero(hessian) == 0


@pytest.mark.unit
def test_prepare_batch_raises_when_no_tokens_contributed():
    module = torch.nn.Linear(4, 3, bias=False)
    hessians = {module: make_empty_hessian(module)}
    num_samples = {module: torch.zeros(())}
    batch_qparams = [
        {
            "scale": torch.ones(3, 1),
            "zero_point": torch.zeros(3, 1, dtype=torch.int8),
            "global_scale": None,
        }
    ]

    with pytest.raises(ValueError, match="No calibration tokens contributed"):
        prepare_batch([module], batch_qparams, hessians, num_samples)


@pytest.mark.unit
def test_select_masked_tokens_flattens_leading_dims():
    activations = torch.arange(24).reshape(2, 3, 4).to(torch.float32)
    loss_mask = torch.tensor([[1, 0, 1], [1, 1, 0]]).to(torch.bool)

    selected = select_masked_tokens(activations, loss_mask, "q_proj")

    expected = activations.reshape(-1, 4)[loss_mask.reshape(-1)]
    assert torch.equal(selected, expected)


@pytest.mark.unit
def test_select_masked_tokens_mismatch_reports_module_name():
    activations = torch.zeros(2, 3, 4)
    loss_mask = torch.zeros(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="q_proj"):
        select_masked_tokens(activations, loss_mask, "q_proj")


@pytest.mark.unit
def test_get_active_loss_mask(monkeypatch):
    class _FakeState:
        loss_masks = [torch.tensor([[1, 0, 1]])]
        current_batch_idx = 0

    class _FakeSession:
        state = _FakeState()

    monkeypatch.setattr("llmcompressor.core.active_session", lambda: _FakeSession())

    loss_mask = get_active_loss_mask()

    assert loss_mask is not None
    assert loss_mask.dtype == torch.bool
    assert loss_mask.tolist() == [[True, False, True]]


@pytest.mark.unit
def test_get_active_loss_mask_out_of_range(monkeypatch):
    class _FakeState:
        loss_masks = [torch.ones(3)]
        current_batch_idx = 5

    class _FakeSession:
        state = _FakeState()

    monkeypatch.setattr("llmcompressor.core.active_session", lambda: _FakeSession())

    assert get_active_loss_mask() is None
