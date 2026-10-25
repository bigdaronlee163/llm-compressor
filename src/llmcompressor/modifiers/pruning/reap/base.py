"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

import json
from functools import partial
from pathlib import Path
from typing import Any, Optional

import torch
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
)
from loguru import logger
from pydantic import Field, PrivateAttr, model_validator
from torch import distributed as dist

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.pruning.reap.utils import (
    MoeModelAttrs,
    REAPSaliencyTracker,
    get_moe_attrs,
    prune_moe_layer,
    update_model_config,
)
from llmcompressor.modifiers.utils.masking import get_active_loss_mask
from llmcompressor.utils.dev import get_main_device

__all__ = ["REAPPruningModifier"]


def _apply_loss_mask(
    top_k_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
    norm_buffer: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """
    Restrict router decisions and expert output norms to masked tokens.

    Called before saliency accumulation so that tokens excluded by the
    calibration ``loss_mask`` do not contribute to the REAP saliency. Returns
    the inputs unchanged when no mask is active.

    :param top_k_indices: ``[num_tokens, top_k]`` selected expert ids
    :param top_k_weights: ``[num_tokens, top_k]`` gate weight per selection
    :param norm_buffer: per-expert output norms, ``[num_tokens]`` per expert
    :return: masked router decisions and norms
    """
    loss_mask = get_active_loss_mask(device=top_k_indices.device)
    if loss_mask is None:
        return top_k_indices, top_k_weights, norm_buffer

    flat_mask = loss_mask.reshape(-1).to(dtype=torch.bool)
    num_tokens = top_k_indices.shape[0]
    if flat_mask.numel() != num_tokens:
        raise ValueError(
            f"Loss mask has {flat_mask.numel()} entries, but the router received "
            f"{num_tokens} tokens. Token masking requires the calibration loss "
            "mask to align with router inputs."
        )

    if get_calibrate_all_experts_flag():
        masked_norms = {
            expert_idx: norms[flat_mask]
            for expert_idx, norms in norm_buffer.items()
            if norms.shape[0] == num_tokens
        }
        if len(masked_norms) != len(norm_buffer):
            raise ValueError(
                "Not every expert recorded output norms for all tokens. REAP "
                "token masking requires all-expert calibration."
            )

        return top_k_indices[flat_mask], top_k_weights[flat_mask], masked_norms

    # Sparse routing: expert norms are ordered by (slot, token), matching
    # top_k_indices.T.reshape(-1) and the torch.where order used when experts
    # are called (see LinearExperts2D.forward)
    flat_idx = top_k_indices.T.reshape(-1)
    token_ids = torch.arange(num_tokens, device=flat_mask.device).repeat(
        top_k_indices.shape[1]
    )
    keep = flat_mask[token_ids]

    masked_norms = {}
    for expert_idx, norms in norm_buffer.items():
        expert_keep = keep[flat_idx == expert_idx]
        if expert_keep.numel() != norms.shape[0]:
            raise ValueError(
                f"Expert {expert_idx} recorded {norms.shape[0]} output norms, but "
                f"{expert_keep.numel()} routed tokens were selected by the loss "
                "mask. REAP token masking requires expert norms to align with "
                "router decisions."
            )
        masked_norms[expert_idx] = norms[expert_keep]

    return flat_idx[keep], top_k_weights.T.reshape(-1)[keep], masked_norms


class REAPPruningModifier(Modifier):
    """
    Prunes experts from MoE layers using the REAP saliency metric. For each
    expert ``j`` the saliency is

        ``S_j = mean(g_j * ||f_j||_2)``

    averaged over the tokens routed to expert ``j``, where:

    - ``g_j`` is the router gate weight assigned to expert ``j`` (the coefficient
      that multiplies the expert's output when combining experts), and
    - ``f_j`` is expert ``j``'s output activation for that token, so
      ``||f_j||_2`` is its L2 norm.

    The lowest-saliency experts are removed per layer. REAP runs during the
    sequential calibration pipeline: saliency is accumulated via hooks on the MoE
    experts, the structural pruning for a layer is executed when it
    completes (``SEQUENTIAL_EPOCH_END``). The config is updated to reflect the new
    number of experts in ``on_finalize``.

    When the calibration dataset provides a ``loss_mask`` and
    ``use_loss_mask=True``, only masked tokens contribute to the saliency. This
    works both with ``moe_calibrate_all_experts=True`` and with sparse routing
    (``moe_calibrate_all_experts=False``).

    :param sparsity: fraction of experts to remove per layer (0, 1).
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the mapping of
        retained expert indices per layer is written after pruning completes.

    Example recipe::

        REAPPruningModifier:
          sparsity: 0.25
    """

    requires_calibration_data: bool = True

    sparsity: float
    ignore: list[str] = Field(default_factory=list)
    report_path: Optional[str] = Field(default=None)

    _moe_attrs: MoeModelAttrs | None = PrivateAttr(default=None)
    _saliency_trackers: dict[str, REAPSaliencyTracker] = PrivateAttr(
        default_factory=dict
    )
    _n_experts_to_drop: int = PrivateAttr(default=0)
    _n_experts_to_drop_per_group: int | None = PrivateAttr(default=None)
    _norm_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )
    _cpu_pg: Any = PrivateAttr(default=None)
    _retained_experts: dict[str, list[int]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sparsity(self) -> "REAPPruningModifier":
        if not 0.0 < self.sparsity < 1.0:
            raise ValueError(f"sparsity must be in (0, 1), got {self.sparsity}")
        return self

    @model_validator(mode="after")
    def _validate_report_path(self) -> "REAPPruningModifier":
        if self.report_path is not None:
            if not self.report_path.endswith(".json"):
                raise ValueError(
                    f"report_path must end with .json, got {self.report_path}"
                )
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
        return self

    def on_initialize(self, state: State, **kwargs) -> bool:
        model = state.model
        self._cpu_pg = dist.new_group(backend="gloo") if is_distributed() else None
        self._moe_attrs = get_moe_attrs(model, self.ignore)
        self._n_experts_to_drop = int(self._moe_attrs.num_experts * self.sparsity)

        if self._n_experts_to_drop == 0:
            raise ValueError(
                f"sparsity={self.sparsity} results in 0 "
                f"experts to drop (out of {self._moe_attrs.num_experts}). "
                "No pruning will be performed."
            )

        if self._moe_attrs.n_group is not None:
            self._n_experts_to_drop_per_group = round(
                self._n_experts_to_drop / self._moe_attrs.n_group
            )

            if self._n_experts_to_drop_per_group == 0:
                raise ValueError(
                    f"Group-limited router detected: sparsity={self.sparsity} "
                    f"results in 0 experts to drop per group "
                    f"(out of {self._moe_attrs.group_size}). "
                    "No pruning will be performed."
                )

        # fail fast (before calibration) if the requested sparsity would
        # leave the router unable to select top_k experts per token
        if self._moe_attrs.n_group is not None:
            retained_per_group = (
                self._moe_attrs.group_size - self._n_experts_to_drop_per_group
            )
            available = self._moe_attrs.top_k_group * retained_per_group
        else:
            available = self._moe_attrs.num_experts - self._n_experts_to_drop

        if self._moe_attrs.top_k > available:
            raise ValueError(
                f"REAP sparsity is too aggressive: the router selects "
                f"top_k={self._moe_attrs.top_k} experts per token, "
                f"but only {available} experts would remain reachable "
                f"after pruning (num_experts={self._moe_attrs.num_experts}, "
                f"dropping≈{self._n_experts_to_drop}). Reduce the sparsity"
            )

        logger.info(
            f"REAP initialized: {len(self._moe_attrs.moe_layer_names)} MoE layers, "
            f"{self._moe_attrs.num_experts} experts/layer, will drop "
            f"{self._n_experts_to_drop} ({self.sparsity:.0%})"
        )
        if self._n_experts_to_drop_per_group is not None:
            logger.info(
                f"Group-limited router detected: will drop "
                f"{self._n_experts_to_drop_per_group} experts/group "
                f"(out of {self._moe_attrs.group_size} in each of "
                f"{self._moe_attrs.n_group} groups)"
            )
            if (
                self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                != self._n_experts_to_drop
            ):
                logger.warning(
                    f"REAP: group-limited routing requires an equal drop "
                    f"per group; dropping "
                    f"{self._n_experts_to_drop_per_group * self._moe_attrs.n_group} "
                    f"experts instead of the requested "
                    f"{self._n_experts_to_drop} "
                    f"(n_group={self._moe_attrs.n_group})"
                )
                self._n_experts_to_drop = (
                    self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                )

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        model = state.model

        # Info if moe_calibrate_all_experts is enabled, which is not necessary for REAP
        if get_calibrate_all_experts_flag():
            logger.info(
                "REAP: moe_calibrate_all_experts is enabled, which is not necessary"
                " for REAP. You can disable it explicity by setting "
                "moe_calibrate_all_experts=False in your oneshot() call. This may "
                "result in faster and/or more memory-efficient calibration, "
                "depending on your model and hardware."
            )

        for layer_name in self._moe_attrs.moe_layer_names:
            module = model.get_submodule(layer_name)

            self._saliency_trackers[layer_name] = REAPSaliencyTracker(
                self._moe_attrs.num_experts
            )
            self._norm_buffers[layer_name] = {}

            # One hook per expert to record its per-token output norm and
            # store in the layer's norm buffer
            experts = getattr(module, self._moe_attrs.experts_attr)
            expert_list = [
                expert for expert in experts.children() if isinstance(expert, ExpertMLP)
            ]  # Filter out the activation function submodule
            for idx, expert in enumerate(expert_list):
                self.register_hook(
                    expert, partial(self._expert_hook, layer_name, idx), "forward"
                )

            # Hook for the experts block to capture the router's top-k
            # routing decisions and weights. This hook also executes pruning,
            # which requires the expert output norms. Therefore, it must be a
            # forward hook, so that it runs after the individual expert hooks
            # have populated the norm buffer
            self.register_hook(
                experts, partial(self._experts_block_hook, layer_name), "forward"
            )

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """Finalize the model config to reflect the new number of experts."""

        model = state.model

        new_num_experts = self._moe_attrs.num_experts - self._n_experts_to_drop
        update_model_config(model, self._moe_attrs, new_num_experts)

        if self.report_path is not None and is_source_process():
            with open(self.report_path, "w") as f:
                json.dump(self._retained_experts, f)

        self._saliency_trackers.clear()
        self._norm_buffers.clear()
        if is_distributed():
            dist.destroy_process_group(group=self._cpu_pg)
            self._cpu_pg = None

        return True

    # -- decision finalization ----------------------------------------------

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """Prune any tracked layer whose saliency is
        complete, then release its activation norm buffers."""

        model = state.model
        expected = self._moe_attrs.num_experts - self._n_experts_to_drop
        device = get_main_device()

        # get trackers which collected samples; sort by layer name to ensure
        # synchonrization across ranks
        trackers = [
            (layer_name, self._saliency_trackers[layer_name])
            for layer_name in sorted(self._saliency_trackers)
            if self._saliency_trackers[layer_name].total_count > 0
        ]

        for layer_name, tracker in trackers:
            tracker.reduce_saliency_stats(device, group=self._cpu_pg)

            if is_source_process():
                retained = tracker.compute_retained_experts(
                    self._n_experts_to_drop,
                    self._n_experts_to_drop_per_group,
                    self._moe_attrs,
                )
            else:
                retained = torch.empty(expected, dtype=torch.int, device="cpu")

            if is_distributed():
                dist.broadcast(retained, src=get_source_rank(), group=self._cpu_pg)

            assert (
                len(retained) == expected
            ), f"Expected {expected} retained experts, got {len(retained)}"

            retained = retained.tolist()
            self._retained_experts[layer_name] = retained
            prune_moe_layer(model, layer_name, retained, self._moe_attrs)

            # free this layer's accumulators / buffers now
            del self._saliency_trackers[layer_name]
            self._norm_buffers.pop(layer_name, None)

    # -- calibration hooks ---------------------------------------------------

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Record expert ``f_j`` output norms for every token this batch."""
        if layer_name not in self._norm_buffers:
            # Layer has already been pruned
            return
        if isinstance(output, tuple):
            output = output[0]
        with torch.no_grad():
            norms = torch.linalg.norm(output.float(), dim=-1).reshape(-1)
        self._norm_buffers[layer_name][expert_idx] = norms

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine router decisions with expert norms to update saliency."""
        top_k_indices = args[1]
        top_k_weights = args[2]

        tracker = self._saliency_trackers.get(layer_name)
        if tracker is None:
            return

        norm_buffer = self._norm_buffers[layer_name]

        top_k_indices, top_k_weights, norm_buffer = _apply_loss_mask(
            top_k_indices, top_k_weights, norm_buffer
        )

        with torch.no_grad():
            tracker.update(top_k_indices, top_k_weights, norm_buffer)

        self._norm_buffers[layer_name] = {}
