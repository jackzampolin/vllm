# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-first LoRA bridge for EXL3 Trellis routed experts."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod


class Exl3TrellisLoRAExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Apply routed LoRA around projection-level EXL3 GEMMs.

    This implementation is intentionally an eager numerical oracle. It keeps
    each EXL3 projection's native input/output rotations inside
    ``Exl3MoEMethod._apply_expert`` and injects LoRA only in the original model
    coordinates:

    ``FC1 base -> FC1 LoRA -> SiLU(gate) * up -> FC2 base -> FC2 LoRA``.

    The no-adapter route dispatches to the existing planned Trellis kernel
    before modular workspaces are allocated. A later Sparkinfer split kernel
    can replace this active-adapter implementation without changing vLLM's
    adapter loading or request mapping.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        quant_method: Exl3MoEMethod,
        layer: RoutedExperts,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.quant_method = quant_method
        self.layer = layer
        self.intermediate_size = moe_config.intermediate_size_per_partition
        self.hidden_size = moe_config.hidden_dim
        self.topk = moe_config.experts_per_token
        self.use_eager_oracle = os.environ.get("VLLM_EXL3_LORA_ORACLE", "0") == "1"
        vllm_config = get_current_vllm_config_or_none()
        experimental_graphs = (
            os.environ.get("VLLM_EXL3_LORA_EXPERIMENTAL_GRAPHS", "0") == "1"
        )
        if (
            vllm_config is not None
            and not vllm_config.model_config.enforce_eager
            and not experimental_graphs
        ):
            raise ValueError(
                "Dynamic EXL3 routed LoRA currently requires --enforce-eager. "
                "CUDA graph adapter switching remains experimental; set "
                "VLLM_EXL3_LORA_EXPERIMENTAL_GRAPHS=1 only for validation."
            )

    def should_use_no_lora_fast_path(self) -> bool:
        """Return whether this forward contains no adapter-backed tokens."""
        context = self._lora_context
        return context is None or context.punica_wrapper.no_lora

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        del weight_key, activation_key
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, global_num_experts, local_num_experts, expert_tokens_meta
        if activation != MoEActivation.SILU:
            raise NotImplementedError(f"EXL3 LoRA supports SiLU only, got {activation}")
        workspace13 = (M, topk, max(2 * self.intermediate_size, K))
        workspace2 = (M * topk, self.intermediate_size)
        output = (M, K)
        return workspace13, workspace2, output

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        del w1, w2
        return (
            self.layer.local_num_experts,
            a1.size(0),
            2 * self.intermediate_size,
            self.hidden_size,
            topk_ids.size(1),
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self.use_eager_oracle:
            self._apply_eager_oracle(
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            return
        del global_num_experts, a1q_scale, a2_scale, workspace13, workspace2
        del expert_tokens_meta
        if activation != MoEActivation.SILU:
            raise NotImplementedError(f"EXL3 LoRA supports SiLU only, got {activation}")
        if expert_map is not None:
            raise NotImplementedError("EXL3 LoRA does not support expert maps")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "EXL3 LoRA does not apply router weights on the input"
            )
        lora_context = self._lora_context
        if lora_context is None:
            raise RuntimeError("EXL3 LoRA context was not initialized")

        binding = self.quant_method.bind_rank_sliced_lora(
            self.layer,
            hidden_states,
            topk_weights,
            topk_ids,
            output=output,
        )
        fc1 = binding.run_fc1_base()
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            lora_context,
            y=fc1,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=None,
            w1=w1,
            w2=w2,
            num_tokens=hidden_states.size(0),
            top_k_num=topk_ids.size(1),
        )
        activated = binding.run_activation()
        route_output = binding.run_fc2_base()
        self.apply_w2_lora(
            lora_context,
            y=route_output,
            x=activated,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=hidden_states.size(0),
            w1=w1,
            w2=w2,
            top_k_num=topk_ids.size(1),
        )
        reduced = binding.run_reduce()
        if reduced.data_ptr() != output.data_ptr():
            output.copy_(reduced)

    def _apply_eager_oracle(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """Run the projection-level numerical oracle for diagnosis."""
        del global_num_experts, a1q_scale, a2_scale, expert_tokens_meta
        if activation != MoEActivation.SILU:
            raise NotImplementedError(f"EXL3 LoRA supports SiLU only, got {activation}")
        if expert_map is not None:
            raise NotImplementedError("EXL3 LoRA does not support expert maps")
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "EXL3 LoRA does not apply router weights on the input"
            )
        lora_context = self._lora_context
        if lora_context is None:
            raise RuntimeError("EXL3 LoRA context was not initialized")

        num_tokens = hidden_states.size(0)
        topk = topk_ids.size(1)
        intermediate_size = self.intermediate_size
        hidden_size = self.hidden_size
        if topk != self.topk:
            raise ValueError(f"expected top-k {self.topk}, got {topk}")

        fc1 = _resize_cache(
            workspace13,
            (num_tokens, topk, 2 * intermediate_size),
        )
        fc1.zero_()
        base_input = hidden_states.to(torch.float16)
        for expert_id in range(self.layer.local_num_experts):
            positions = (topk_ids == expert_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            route_ids = positions[:, 1]
            expert_input = base_input.index_select(0, token_ids)
            gate = self.quant_method._apply_expert(
                self.layer, "w13", expert_input, expert_id, "w1"
            )
            up = self.quant_method._apply_expert(
                self.layer, "w13", expert_input, expert_id, "w3"
            )
            fc1[token_ids, route_ids, :intermediate_size] = gate.to(fc1.dtype)
            fc1[token_ids, route_ids, intermediate_size:] = up.to(fc1.dtype)

        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            lora_context,
            y=fc1,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=None,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=topk,
        )

        activated = _resize_cache(
            workspace2,
            (num_tokens * topk, intermediate_size),
        )
        gate = fc1[..., :intermediate_size].view(-1, intermediate_size)
        up = fc1[..., intermediate_size:].view(-1, intermediate_size)
        torch.mul(F.silu(gate), up, out=activated)

        route_output = _resize_cache(
            workspace13,
            (num_tokens, topk, hidden_size),
        )
        route_output.zero_()
        activated_routes = activated.view(num_tokens, topk, intermediate_size)
        for expert_id in range(self.layer.local_num_experts):
            positions = (topk_ids == expert_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            token_ids = positions[:, 0]
            route_ids = positions[:, 1]
            expert_input = activated_routes[token_ids, route_ids].to(torch.float16)
            expert_output = self.quant_method._apply_expert(
                self.layer, "w2", expert_input, expert_id, "w2"
            )
            route_weight = topk_weights[token_ids, route_ids].unsqueeze(-1)
            route_output[token_ids, route_ids] = (expert_output * route_weight).to(
                route_output.dtype
            )

        self.apply_w2_lora(
            lora_context,
            y=route_output,
            x=activated,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=topk,
        )
        # FusedMoEKernel may alias ``output`` to workspace13, which currently
        # stores route_output. Materialize the reduction before copying so the
        # source is never overwritten while it is still being read.
        output.copy_(route_output.sum(dim=1))


__all__ = ["Exl3TrellisLoRAExperts"]
