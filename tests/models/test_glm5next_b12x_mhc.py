# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.glm5next.nvidia import b12x_mhc


def test_b12x_mhc_adapter_uses_public_plan_bind_run(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )

    def bind(bound_plan, **kwargs):
        calls["bind"] = (bound_plan, kwargs)
        return SimpleNamespace(**kwargs)

    def run_pre(*args, **kwargs):
        calls["pre"] = (args, kwargs)
        binding = kwargs["binding"]
        return binding.out, binding.post, binding.comb, binding.y

    def run_post_pre(*args, **kwargs):
        calls["post_pre"] = (args, kwargs)
        binding = kwargs["binding"]
        return binding.out, binding.post, binding.comb, binding.y

    def run_post(*args):
        calls["post"] = args
        return args[1]

    module = SimpleNamespace(
        Caps=make_caps,
        DEFAULT_BLOCK_K=128,
        MULT=4,
        bind=bind,
        is_supported=lambda: True,
        plan=lambda caps: plan,
        run_post=run_post,
        run_post_pre=run_post_pre,
        run_pre=run_pre,
    )
    workspace = SimpleNamespace(
        get_simultaneous=lambda *args: [torch.empty(args[0][0], dtype=args[0][1])]
    )
    monkeypatch.setattr(b12x_mhc, "get_b12x_mhc", lambda: module)
    monkeypatch.setattr(b12x_mhc, "current_workspace_manager", lambda: workspace)

    mhc = b12x_mhc.B12xMHCResidual(
        hidden_size=256,
        hc_mult=4,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    residual = torch.empty((3, 256), dtype=torch.bfloat16)
    hc_fn = torch.empty((24, 256), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)
    norm_weight = torch.empty((256,), dtype=torch.bfloat16)

    residual_out, post, comb, layer_input = mhc.run_pre(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    next_outputs = mhc.run_post_pre(
        layer_input,
        residual_out,
        post,
        comb,
        torch.empty((24, 1024), dtype=torch.float32),
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    final = mhc.run_post(layer_input, *next_outputs[:3])

    caps = calls["caps"]
    assert isinstance(caps, dict)
    assert caps["hidden_size"] == 256
    assert caps["split_k"] == 8
    bind_call = calls["bind"]
    assert isinstance(bind_call, tuple)
    assert bind_call[1]["scratch"].dtype == torch.uint8
    assert residual_out.shape == (3, 4, 256)
    assert layer_input.shape == (3, 256)
    assert final is next_outputs[0]
