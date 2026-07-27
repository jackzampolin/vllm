#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit a Macaron GLM-5.2 adapter without downloading its tensor payload."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

EXPECTED_REPO = "mindlab-research/Macaron-V1-Venti"
EXPECTED_REVISION = "3d6f30eea38663a7b9320f3a6b28822ed4aa7ac4"
EXPECTED_SUBFOLDER = "loras/L2"
EXPECTED_PAYLOAD_BYTES = 15_393_360_208
EXPECTED_TARGETS = {
    "down_proj",
    "gate_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "q_a_proj",
    "q_b_proj",
    "up_proj",
}
ATTENTION_PROJECTIONS = {
    "kv_a_proj_with_mqa": ((16, 6144), (576, 16)),
    "kv_b_proj": ((16, 512), (28672, 16)),
    "o_proj": ((16, 16384), (6144, 16)),
    "q_a_proj": ((16, 6144), (2048, 16)),
    "q_b_proj": ((16, 2048), (16384, 16)),
}
DENSE_MLP_PROJECTIONS = {
    "down_proj": ((16, 12288), (6144, 16)),
    "gate_proj": ((16, 6144), (12288, 16)),
    "up_proj": ((16, 6144), (12288, 16)),
}
EXPERT_MLP_PROJECTIONS = {
    "down_proj": ((16, 2048), (6144, 16)),
    "gate_proj": ((16, 6144), (2048, 16)),
    "up_proj": ((16, 6144), (2048, 16)),
}


def _tensor_key(prefix: str, projection: str, factor: str) -> str:
    return f"{prefix}.{projection}.lora_{factor}.weight"


def expected_l2_tensors() -> dict[str, tuple[int, ...]]:
    """Return the exact Macaron L2 tensor-name and shape contract."""
    result: dict[str, tuple[int, ...]] = {}
    prefix = "base_model.model.model.layers"
    for layer in range(78):
        layer_prefix = f"{prefix}.{layer}"
        for projection, shapes in ATTENTION_PROJECTIONS.items():
            result[_tensor_key(f"{layer_prefix}.self_attn", projection, "A")] = shapes[
                0
            ]
            result[_tensor_key(f"{layer_prefix}.self_attn", projection, "B")] = shapes[
                1
            ]

        if layer < 3:
            for projection, shapes in DENSE_MLP_PROJECTIONS.items():
                result[_tensor_key(f"{layer_prefix}.mlp", projection, "A")] = shapes[0]
                result[_tensor_key(f"{layer_prefix}.mlp", projection, "B")] = shapes[1]
            continue

        for expert in range(256):
            expert_prefix = f"{layer_prefix}.mlp.experts.{expert}"
            for projection, shapes in EXPERT_MLP_PROJECTIONS.items():
                result[_tensor_key(expert_prefix, projection, "A")] = shapes[0]
                result[_tensor_key(expert_prefix, projection, "B")] = shapes[1]
        shared_prefix = f"{layer_prefix}.mlp.shared_experts"
        for projection, shapes in EXPERT_MLP_PROJECTIONS.items():
            result[_tensor_key(shared_prefix, projection, "A")] = shapes[0]
            result[_tensor_key(shared_prefix, projection, "B")] = shapes[1]
    return result


def validate_adapter_config(config: dict[str, Any]) -> None:
    """Fail unless the PEFT config matches the supported L2 contract."""
    expected_scalars = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
    }
    failures = [
        f"{name}: expected {expected!r}, got {config.get(name)!r}"
        for name, expected in expected_scalars.items()
        if config.get(name) != expected
    ]
    targets = set(config.get("target_modules", ()))
    if targets != EXPECTED_TARGETS:
        failures.append(
            "target_modules: expected "
            f"{sorted(EXPECTED_TARGETS)!r}, got {sorted(targets)!r}"
        )
    if failures:
        raise ValueError("unsupported Macaron adapter config:\n" + "\n".join(failures))


def validate_safetensors_header(
    header: dict[str, Any],
    *,
    file_size: int,
    header_size: int,
) -> dict[str, Any]:
    """Validate exact tensor coverage, shapes, dtypes, and data offsets."""
    expected = expected_l2_tensors()
    actual = {key: value for key, value in header.items() if key != "__metadata__"}
    missing = sorted(expected.keys() - actual.keys())
    unknown = sorted(actual.keys() - expected.keys())
    if missing or unknown:
        raise ValueError(
            "Macaron L2 tensor coverage mismatch: "
            f"missing={missing[:8]!r} ({len(missing)} total), "
            f"unknown={unknown[:8]!r} ({len(unknown)} total)"
        )

    payload_bytes = file_size - 8 - header_size
    ranges: list[tuple[int, int, str]] = []
    dtype_counts: Counter[str] = Counter()
    for name, shape in expected.items():
        entry = actual[name]
        dtype = entry.get("dtype")
        actual_shape = tuple(entry.get("shape", ()))
        offsets = tuple(entry.get("data_offsets", ()))
        if dtype != "BF16":
            raise TypeError(f"{name}: expected BF16, got {dtype!r}")
        if actual_shape != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {actual_shape}")
        if len(offsets) != 2:
            raise ValueError(f"{name}: invalid data_offsets {offsets!r}")
        start, end = (int(value) for value in offsets)
        expected_bytes = math.prod(shape) * 2
        if start < 0 or end - start != expected_bytes or end > payload_bytes:
            raise ValueError(
                f"{name}: invalid byte range {(start, end)}, "
                f"expected {expected_bytes} bytes inside {payload_bytes}"
            )
        ranges.append((start, end, name))
        dtype_counts[dtype] += 1

    ranges.sort()
    cursor = 0
    for start, end, name in ranges:
        if start != cursor:
            raise ValueError(
                f"{name}: non-contiguous or overlapping payload at {start}; "
                f"expected {cursor}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise ValueError(
            f"tensor ranges cover {cursor} bytes, payload has {payload_bytes}"
        )

    return {
        "tensor_count": len(actual),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "payload_bytes": payload_bytes,
        "layers": [0, 77],
        "routed_expert_layers": [3, 77],
        "experts_per_layer": 256,
        "mtp_layer_78_present": False,
    }


def _get_range(url: str, start: int, end: int) -> bytes:
    headers = build_hf_headers(token=True)
    headers["Range"] = f"bytes={start}-{end}"
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(
            f"server ignored byte range {start}-{end}: HTTP {response.status_code}"
        )
    expected_length = end - start + 1
    if len(response.content) != expected_length:
        raise RuntimeError(
            f"short byte range {start}-{end}: got {len(response.content)} bytes"
        )
    return response.content


def inspect_remote_adapter(
    repo_id: str,
    revision: str,
    subfolder: str,
) -> dict[str, Any]:
    """Fetch only config and safetensors header, then return an audit report."""
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    resolved_revision = info.sha
    if (
        repo_id == EXPECTED_REPO
        and subfolder == EXPECTED_SUBFOLDER
        and resolved_revision != EXPECTED_REVISION
    ):
        raise ValueError(
            "Macaron L2 revision drift: "
            f"expected {EXPECTED_REVISION}, got {resolved_revision}"
        )
    config_name = f"{subfolder}/adapter_config.json"
    tensor_name = f"{subfolder}/adapter_model.safetensors"
    sizes = {
        sibling.rfilename: sibling.size
        for sibling in info.siblings
        if sibling.rfilename in {config_name, tensor_name}
    }
    if set(sizes) != {config_name, tensor_name}:
        raise FileNotFoundError(
            f"{repo_id}@{resolved_revision} is missing L2 adapter files"
        )
    file_size = int(sizes[tensor_name] or 0)
    if (
        repo_id == EXPECTED_REPO
        and resolved_revision == EXPECTED_REVISION
        and subfolder == EXPECTED_SUBFOLDER
        and file_size != EXPECTED_PAYLOAD_BYTES
    ):
        raise ValueError(
            f"pinned L2 file size changed: {file_size} != {EXPECTED_PAYLOAD_BYTES}"
        )

    config_url = hf_hub_url(repo_id, config_name, revision=resolved_revision)
    config_response = requests.get(
        config_url,
        headers=build_hf_headers(token=True),
        timeout=30,
    )
    config_response.raise_for_status()
    config = config_response.json()
    validate_adapter_config(config)

    tensor_url = hf_hub_url(repo_id, tensor_name, revision=resolved_revision)
    header_size = struct.unpack("<Q", _get_range(tensor_url, 0, 7))[0]
    if header_size <= 2 or header_size >= file_size - 8:
        raise ValueError(f"invalid safetensors header size {header_size}")
    header = json.loads(_get_range(tensor_url, 8, 7 + header_size))
    tensor_report = validate_safetensors_header(
        header,
        file_size=file_size,
        header_size=header_size,
    )
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "subfolder": subfolder,
        "file_size": file_size,
        "header_size": header_size,
        "config": {
            "rank": config["r"],
            "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"],
            "bias": config["bias"],
            "targets": sorted(config["target_modules"]),
        },
        **tensor_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=EXPECTED_REPO)
    parser.add_argument("--revision", default=EXPECTED_REVISION)
    parser.add_argument("--subfolder", default=EXPECTED_SUBFOLDER)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON report to this path after validation succeeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_remote_adapter(args.repo_id, args.revision, args.subfolder)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
