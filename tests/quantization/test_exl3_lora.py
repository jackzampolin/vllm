# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod
from types import MethodType, SimpleNamespace

import pytest
import torch

from tools.exl3_lora.inspect_adapter import (
    expected_l2_tensors,
    validate_safetensors_header,
)
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import is_in_target_modules, parse_fine_tuned_lora_name
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.exl3_trellis import (
    Exl3TrellisLoRAExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def test_exl3_lora_oracle_preserves_projection_order() -> None:
    hidden_size = 4
    intermediate_size = 3
    num_experts = 2
    topk = 2

    base = {
        "w1": torch.tensor(
            [
                [[0.2, -0.1, 0.3, 0.4], [0.5, 0.2, -0.2, 0.1], [0.3] * 4],
                [[-0.2, 0.4, 0.1, 0.3], [0.1, -0.3, 0.5, 0.2], [0.2] * 4],
            ]
        ),
        "w3": torch.tensor(
            [
                [[0.3, 0.1, -0.4, 0.2], [0.2, 0.5, 0.1, -0.2], [0.4] * 4],
                [[0.4, -0.2, 0.3, 0.1], [-0.1, 0.2, 0.4, 0.3], [0.1] * 4],
            ]
        ),
        "w2": torch.tensor(
            [
                [[0.2, 0.1, 0.3], [0.4, -0.2, 0.1], [0.1, 0.5, -0.3], [0.2] * 3],
                [[-0.1, 0.3, 0.2], [0.5, 0.1, -0.2], [0.2, -0.4, 0.3], [0.3] * 3],
            ]
        ),
    }
    delta = {
        "w1": torch.arange(24, dtype=torch.float32).view(2, 3, 4) / 200,
        "w3": torch.arange(24, 48, dtype=torch.float32).view(2, 3, 4) / 250,
        "w2": torch.arange(24, dtype=torch.float32).view(2, 4, 3) / 300,
    }

    class FakeQuantMethod:
        @staticmethod
        def _apply_expert(layer, group, x, expert_id, shard_id):
            del layer, group
            return x @ base[shard_id][expert_id].to(x.dtype).T

    experts = object.__new__(Exl3TrellisLoRAExperts)
    experts.quant_method = FakeQuantMethod()
    experts.layer = SimpleNamespace(local_num_experts=num_experts)
    experts.intermediate_size = intermediate_size
    experts.hidden_size = hidden_size
    experts.topk = topk
    experts._lora_context = SimpleNamespace()
    experts.use_eager_oracle = True

    def apply_w13_lora(
        self,
        context,
        *,
        y,
        x,
        topk_ids,
        **kwargs,
    ):
        del self, context, kwargs
        for token in range(x.size(0)):
            for route in range(topk_ids.size(1)):
                expert = int(topk_ids[token, route])
                y[token, route, :intermediate_size].add_(
                    x[token] @ delta["w1"][expert].T
                )
                y[token, route, intermediate_size:].add_(
                    x[token] @ delta["w3"][expert].T
                )
        return None, None, None, None

    def apply_w2_lora(
        self,
        context,
        *,
        y,
        x,
        topk_weights,
        num_tokens,
        top_k_num,
        **kwargs,
    ):
        del self, context, kwargs
        x = x.view(num_tokens, top_k_num, intermediate_size)
        for token in range(num_tokens):
            for route in range(top_k_num):
                expert = int(topk_ids[token, route])
                y[token, route].add_(
                    (x[token, route] @ delta["w2"][expert].T)
                    * topk_weights[token, route]
                )

    experts.apply_w13_lora = MethodType(apply_w13_lora, experts)
    experts.apply_w2_lora = MethodType(apply_w2_lora, experts)

    hidden_states = torch.tensor(
        [[0.5, -0.3, 0.7, 0.2], [-0.4, 0.6, 0.1, 0.8]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    topk_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)

    common = torch.empty(
        2 * topk * max(2 * intermediate_size, hidden_size),
        dtype=torch.float32,
    )
    workspace13 = common.view(2, topk, -1)
    output = common[: 2 * hidden_size].view(2, hidden_size)
    workspace2 = torch.empty(2 * topk, intermediate_size)
    logical_w13 = torch.empty(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="meta",
    )
    logical_w2 = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size,
        device="meta",
    )

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=logical_w13,
        w2=logical_w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    expected = torch.zeros_like(output)
    for token in range(hidden_states.size(0)):
        for route in range(topk):
            expert = int(topk_ids[token, route])
            base_x = hidden_states[token].to(torch.float16)
            gate = (base_x @ base["w1"][expert].to(torch.float16).T).to(torch.float32)
            up = (base_x @ base["w3"][expert].to(torch.float16).T).to(torch.float32)
            gate.add_(hidden_states[token] @ delta["w1"][expert].T)
            up.add_(hidden_states[token] @ delta["w3"][expert].T)
            activated = torch.nn.functional.silu(gate) * up
            down = (
                activated.to(torch.float16) @ base["w2"][expert].to(torch.float16).T
            ).to(torch.float32)
            down.add_(activated @ delta["w2"][expert].T)
            expected[token].add_(down * topk_weights[token, route])

    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


def test_exl3_no_lora_uses_original_quant_method() -> None:
    sentinel = torch.tensor([[17.0]])

    class OldMethod:
        def apply(self, *args):
            self.args = args
            return sentinel

    class Kernel:
        class Experts:
            @staticmethod
            def should_use_no_lora_fast_path():
                return True

        fused_experts = Experts()

        @staticmethod
        def apply(**kwargs):
            raise AssertionError(f"modular kernel unexpectedly called: {kwargs}")

    method = object.__new__(FusedMoEModularMethod)
    method.moe_kernel = Kernel()
    method.old_quant_method = OldMethod()

    layer = object()
    x = torch.tensor([[1.0]])
    topk_weights = torch.tensor([[1.0]])
    topk_ids = torch.tensor([[0]])
    result = method.apply(layer, x, topk_weights, topk_ids, None, None)

    assert result is sentinel
    assert method.old_quant_method.args == (
        layer,
        x,
        topk_weights,
        topk_ids,
        None,
        None,
    )


def test_exl3_missing_lora_context_uses_base_path() -> None:
    experts = object.__new__(Exl3TrellisLoRAExperts)
    experts._lora_context = None
    assert experts.should_use_no_lora_fast_path()


def test_exl3_planned_split_injects_lora_between_base_stages() -> None:
    events: list[str] = []
    hidden_states = torch.tensor([[1.0, 2.0]])
    topk_ids = torch.tensor([[0]])
    topk_weights = torch.tensor([[1.0]])
    output = torch.empty_like(hidden_states)
    fc1 = torch.zeros((1, 1, 4))
    activation = torch.full((1, 2), 3.0)
    route_output = torch.zeros((1, 1, 2))

    class Binding:
        @staticmethod
        def run_fc1_base():
            events.append("fc1_base")
            return fc1

        @staticmethod
        def run_activation():
            assert torch.equal(fc1, torch.ones_like(fc1))
            events.append("activation")
            return activation

        @staticmethod
        def run_fc2_base():
            events.append("fc2_base")
            return route_output

        @staticmethod
        def run_reduce():
            assert torch.equal(route_output, torch.full_like(route_output, 2.0))
            events.append("reduce")
            output.copy_(route_output.sum(dim=1))
            return output

    class QuantMethod:
        @staticmethod
        def bind_rank_sliced_lora(*args, **kwargs):
            del args, kwargs
            events.append("bind")
            return Binding()

    experts = object.__new__(Exl3TrellisLoRAExperts)
    experts.quant_method = QuantMethod()
    experts.layer = object()
    experts.use_eager_oracle = False
    experts._lora_context = SimpleNamespace()

    def apply_w13_lora(self, context, *, y, **kwargs):
        del self, context, kwargs
        events.append("fc1_lora")
        y.add_(1)
        return "sorted", "experts", "count", "mapping"

    def apply_w2_lora(self, context, *, y, x, **kwargs):
        del self, context, kwargs
        assert x is activation
        events.append("fc2_lora")
        y.add_(2)

    experts.apply_w13_lora = MethodType(apply_w13_lora, experts)
    experts.apply_w2_lora = MethodType(apply_w2_lora, experts)
    logical_w13 = torch.empty((1, 4, 2), device="meta")
    logical_w2 = torch.empty((1, 2, 2), device="meta")
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=logical_w13,
        w2=logical_w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=1,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=torch.empty(1),
        workspace2=torch.empty(1),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert events == [
        "bind",
        "fc1_base",
        "fc1_lora",
        "activation",
        "fc2_base",
        "fc2_lora",
        "reduce",
    ]
    torch.testing.assert_close(output, torch.full_like(output, 2.0))


def test_macaron_l2_inventory_is_exact_and_fail_closed() -> None:
    expected = expected_l2_tensors()
    assert len(expected) == 116448
    payload_cursor = 0
    header = {}
    for name, shape in expected.items():
        tensor_bytes = prod(shape) * 2
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [payload_cursor, payload_cursor + tensor_bytes],
        }
        payload_cursor += tensor_bytes

    report = validate_safetensors_header(
        header,
        file_size=8 + 1024 + payload_cursor,
        header_size=1024,
    )
    assert report["tensor_count"] == 116448
    assert report["payload_bytes"] == payload_cursor
    assert report["mtp_layer_78_present"] is False

    removed_name, removed = header.popitem()
    try:
        validate_safetensors_header(
            header,
            file_size=8 + 1024 + payload_cursor,
            header_size=1024,
        )
    except ValueError as exc:
        assert removed_name in str(exc)
    else:
        raise AssertionError(f"missing tensor {removed_name} was accepted")
    finally:
        header[removed_name] = removed


def test_macaron_l2_names_match_glm_runtime_packing() -> None:
    packed_modules = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    target_modules = [
        "down_proj",
        "gate_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "up_proj",
    ]
    for runtime_module in (
        "model.layers.0.self_attn.fused_qkv_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.3.mlp.shared_experts.gate_up_proj",
        "model.layers.3.mlp.shared_experts.down_proj",
    ):
        assert is_in_target_modules(runtime_module, target_modules, packed_modules)

    expert_mapping = RoutedExperts.build_expert_params_mapping(
        "gate_proj",
        "down_proj",
        "up_proj",
        num_experts=256,
        routed_experts_prefix="",
    )
    expert_children = {
        weight_name.rstrip(".") for _, weight_name, _, _ in expert_mapping
    }
    assert len(expert_children) == 256 * 3

    expected = expected_l2_tensors()
    parsed_modules = {
        parse_fine_tuned_lora_name(tensor_name)[0] for tensor_name in expected
    }
    for layer in range(3, 78):
        layer_prefix = f"model.layers.{layer}.mlp."
        layer_experts = {
            module.removeprefix(layer_prefix)
            for module in parsed_modules
            if module.startswith(f"{layer_prefix}experts.")
        }
        assert layer_experts == expert_children


def test_lora_loader_rejects_incomplete_factor_pair() -> None:
    helper = PEFTHelper(
        r=16,
        lora_alpha=32,
        target_modules=["gate_proj"],
    )
    tensors = {
        "base_model.model.model.layers.3.mlp.experts.0.gate_proj."
        "lora_A.weight": torch.ones(16, 4)
    }
    with pytest.raises(ValueError, match="exactly one A tensor and one B tensor"):
        LoRAModel.from_lora_tensors(
            lora_model_id=1,
            tensors=tensors,
            peft_helper=helper,
            device="meta",
        )


def test_fused_moe_accepts_load_time_tp_slices() -> None:
    wrapper = object.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(wrapper)
    wrapper.tp_size = 4
    wrapper.tp_rank = 2
    wrapper.fully_sharded = True
    wrapper.moe_config = SimpleNamespace(intermediate_size_per_partition=5)
    wrapper.w13_lora_a_stacked = (torch.empty(1, 2, 4, 12),)
    wrapper.w2_lora_b_stacked = (torch.empty(1, 2, 3, 16),)

    w13_a = torch.arange(2 * 16 * 12).reshape(2, 16, 12)
    w13_b = torch.arange(2 * 20 * 16).reshape(2, 20, 16)
    w2_a = torch.arange(2 * 16 * 20).reshape(2, 16, 20)
    w2_b = torch.arange(2 * 12 * 16).reshape(2, 12, 16)

    expected = (
        w13_a[:, 8:12],
        w13_b[:, 10:15],
        w2_a[:, :, 10:15],
        w2_b[:, 6:9],
    )
    actual_from_full = (
        wrapper._slice_w13_a(w13_a),
        wrapper._slice_w13_b(w13_b),
        wrapper._slice_w2_a(w2_a),
        wrapper._slice_w2_b(w2_b),
    )
    actual_from_local = (
        wrapper._slice_w13_a(expected[0]),
        wrapper._slice_w13_b(expected[1]),
        wrapper._slice_w2_a(expected[2]),
        wrapper._slice_w2_b(expected[3]),
    )
    for full_result, local_result, expected_result in zip(
        actual_from_full, actual_from_local, expected, strict=True
    ):
        torch.testing.assert_close(full_result, expected_result)
        torch.testing.assert_close(local_result, expected_result)
