# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X mHC residual adapter used by GLM5Next."""

from typing import Any

import torch

from vllm.utils.b12x import get_b12x_mhc
from vllm.v1.worker.workspace import current_workspace_manager


def _require_b12x_mhc() -> Any:
    module = get_b12x_mhc()
    if module is None:
        raise RuntimeError("GLM5Next B12X mHC requires `pip install vllm[b12x]`.")
    if not module.is_supported():
        raise RuntimeError("B12X mHC is not supported on this device.")
    return module


class B12xMHCResidual:
    """Adapt the public B12X mHC plan/bind/run API to GLM's residual path."""

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_mult: int,
        rms_eps: float,
        hc_eps: float,
        sinkhorn_iters: int,
    ) -> None:
        module = _require_b12x_mhc()
        self._caps = module.Caps
        self._plan = module.plan
        self._bind = module.bind
        self._run_pre = module.run_pre
        self._run_post = module.run_post
        self._run_post_pre = module.run_post_pre

        expected_hc_mult = int(module.MULT)
        if hc_mult != expected_hc_mult:
            raise NotImplementedError(
                f"B12X mHC requires hc_mult={expected_hc_mult}, got {hc_mult}."
            )

        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.block_k = int(module.DEFAULT_BLOCK_K)
        total_k = self.hc_mult * self.hidden_size
        if total_k % self.block_k != 0:
            raise ValueError(
                "B12X mHC requires hc_mult * hidden_size to be divisible by "
                f"block_k={self.block_k}, got {total_k}."
            )
        self.split_k = total_k // self.block_k

    def _binding(
        self,
        x: torch.Tensor,
        *,
        expected_m: int,
        y: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> Any:
        tokens = int(x.shape[0])
        expected_m = int(expected_m)
        plan = self._plan(
            self._caps(
                device=x.device,
                dtype=x.dtype,
                max_tokens=max(1, tokens, expected_m),
                hidden_size=self.hidden_size,
                split_k=self.split_k,
            )
        )
        buffers = current_workspace_manager().get_simultaneous(
            *plan.shapes_and_dtypes()
        )
        if not buffers:
            raise ValueError("B12X mHC scratch plan did not provide any buffers.")
        scratch: torch.Tensor | tuple[torch.Tensor, ...]
        scratch = buffers[0] if len(buffers) == 1 else tuple(buffers)
        return self._bind(
            plan,
            scratch=scratch,
            tokens=tokens,
            y=y,
            post=post,
            comb=comb,
            out=out,
            expected_m=expected_m,
        )

    def run_pre(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {
            "rms_eps": self.rms_eps,
            "hc_eps": self.hc_eps,
            "sinkhorn_iters": self.sinkhorn_iters,
            "norm_weight": norm_weight,
            "norm_eps": float(norm_eps),
            "block_k": self.block_k,
        }
        if torch.compiler.is_compiling():
            return self._run_pre(
                residual,
                hc_fn,
                hc_scale,
                hc_base,
                split_k=self.split_k,
                **kwargs,
            )

        tokens, hidden_size = residual.shape
        residual_out = torch.empty(
            (tokens, self.hc_mult, hidden_size),
            dtype=residual.dtype,
            device=residual.device,
        )
        layer_input = torch.empty(
            (tokens, hidden_size), dtype=residual.dtype, device=residual.device
        )
        post_mix = torch.empty(
            (tokens, self.hc_mult), dtype=torch.float32, device=residual.device
        )
        res_mix = torch.empty(
            (tokens, self.hc_mult, self.hc_mult),
            dtype=torch.float32,
            device=residual.device,
        )
        binding = self._binding(
            residual,
            expected_m=int(tokens),
            y=layer_input,
            post=post_mix,
            comb=res_mix,
            out=residual_out,
        )
        return self._run_pre(
            residual,
            hc_fn,
            hc_scale,
            hc_base,
            binding=binding,
            **kwargs,
        )

    def run_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        *,
        norm_weight: torch.Tensor,
        norm_eps: float,
        hc_fn_bf16: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_m = int(residual.shape[0])
        kwargs = {
            "rms_eps": self.rms_eps,
            "hc_eps": self.hc_eps,
            "sinkhorn_iters": self.sinkhorn_iters,
            "norm_weight": norm_weight,
            "norm_eps": float(norm_eps),
            "block_k": self.block_k,
            "expected_m": expected_m,
            "fn_bf16": hc_fn_bf16,
        }
        if torch.compiler.is_compiling():
            return self._run_post_pre(
                x,
                residual,
                post,
                comb,
                hc_fn,
                hc_scale,
                hc_base,
                split_k=self.split_k,
                **kwargs,
            )

        tokens, hc_mult, hidden_size = residual.shape
        residual_out = torch.empty_like(residual)
        layer_input = torch.empty(
            (tokens, hidden_size), dtype=residual.dtype, device=residual.device
        )
        post_out = torch.empty(
            (tokens, hc_mult), dtype=torch.float32, device=residual.device
        )
        comb_out = torch.empty(
            (tokens, hc_mult, hc_mult), dtype=torch.float32, device=residual.device
        )
        binding = self._binding(
            residual,
            expected_m=expected_m,
            y=layer_input,
            post=post_out,
            comb=comb_out,
            out=residual_out,
        )
        return self._run_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            binding=binding,
            **kwargs,
        )

    def run_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_post(x, residual, post, comb)


__all__ = ["B12xMHCResidual"]
