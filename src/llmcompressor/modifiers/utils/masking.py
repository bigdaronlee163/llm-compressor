"""
Utilities for restricting calibration statistics to a subset of tokens.

Modifiers and observers may be calibrated on data which contains tokens that
should not contribute to their statistics, such as user prompts in
instruction-tuned or reasoning datasets. Datasets express this with a
per-token ``loss_mask`` (1 = include, 0 = exclude), which pipelines store on
the active session state. These helpers translate that state into token
selection utilities usable by any modifier or observer.
"""

import torch
from loguru import logger

__all__ = [
    "get_active_loss_mask",
    "select_masked_tokens",
    "mask_activations_with_loss_mask",
]


def get_active_loss_mask(
    device: torch.device | None = None,
    state=None,
) -> torch.Tensor | None:
    """
    Fetch the loss mask belonging to the batch currently being calibrated.

    Pipelines populate ``State.loss_masks`` when
    ``DatasetArguments.use_loss_mask`` is enabled and update
    ``State.current_batch_idx`` before each forward pass. Modifiers and
    observers can call this helper from their calibration hooks to retrieve the
    mask for the current batch.

    :param device: optional device to move the mask to
    :param state: optional state to read from, defaults to the active session's
        state
    :return: boolean mask for the current batch, or None if no mask is
        available
    """
    if state is None:
        # imported lazily to avoid a circular import through llmcompressor.core
        from llmcompressor.core import active_session

        state = getattr(active_session(), "state", None)

    if state is None or state.loss_masks is None:
        return None

    batch_idx = state.current_batch_idx
    if batch_idx < 0 or batch_idx >= len(state.loss_masks):
        return None

    loss_mask = state.loss_masks[batch_idx]
    if loss_mask is None:
        return None

    loss_mask = loss_mask.to(dtype=torch.bool)
    if device is not None:
        loss_mask = loss_mask.to(device=device)

    return loss_mask


def select_masked_tokens(
    activations: torch.Tensor,
    loss_mask: torch.Tensor,
    module_name: str = "module",
    strict: bool = True,
) -> torch.Tensor:
    """
    Keep only the tokens of activations which are enabled by the loss mask.

    Activations are flattened to ``[tokens, hidden]`` before selection, so any
    leading dimensions (batch, sequence, etc.) are supported.

    :param activations: activations of shape ``[..., hidden]``
    :param loss_mask: per-token mask of shape ``[...,]`` whose number of
        elements must match the number of tokens in activations
    :param module_name: name of the module being calibrated, used in errors
    :param strict: whether to raise when the mask cannot be aligned with the
        activations. When False, the activations are returned unchanged
    :return: activations of shape ``[masked_tokens, hidden]``
    """
    hidden_size = activations.shape[-1]
    tokens = activations.reshape(-1, hidden_size)

    flat_mask = loss_mask.reshape(-1).to(device=tokens.device, dtype=torch.bool)
    if flat_mask.numel() != tokens.shape[0]:
        message = (
            f"Loss mask has {flat_mask.numel()} entries, but {module_name} received "
            f"{tokens.shape[0]} tokens. Token masking requires the calibration loss "
            "mask to align with module inputs. This can happen with MoE models when "
            "`moe_calibrate_all_experts=False`, since experts only see routed tokens, "
            "or with inputs which are not token-aligned."
        )
        if strict:
            raise ValueError(message)

        logger.warning(f"{message} Skipping token masking.", log_once=True)
        return activations

    return tokens[flat_mask]


def mask_activations_with_loss_mask(
    activations: torch.Tensor,
    module_name: str = "module",
    strict: bool = False,
) -> torch.Tensor:
    """
    Apply the loss mask of the current batch to an observed activation tensor.

    Convenience wrapper around :func:`get_active_loss_mask` and
    :func:`select_masked_tokens` for observers which do not need to inspect the
    mask themselves.

    :param activations: observed activations of shape ``[..., hidden]``
    :param module_name: name of the module being calibrated, used in errors
    :param strict: whether to raise when the mask cannot be aligned with the
        activations
    :return: masked activations, or the original activations when no mask is
        active
    """
    loss_mask = get_active_loss_mask(device=activations.device)
    if loss_mask is None:
        return activations

    return select_masked_tokens(
        activations, loss_mask, module_name=module_name, strict=strict
    )
