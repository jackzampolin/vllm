# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_LAUNCHER = _REPO_ROOT / "serve-ds4-flash.sh"


def _dry_run(tmp_path: Path, **overrides: str) -> str:
    """Run the DS4 launcher without starting a server.

    Args:
        tmp_path: Isolated home and cache root for the launcher.
        **overrides: Environment values applied to the launcher defaults.

    Returns:
        The launcher's diagnostic stderr output.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "1",
        "MODE": "mtp0",
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr


def _run_with_fake_calibration(
    tmp_path: Path,
    *,
    calibration_exit: int = 0,
    **overrides: str,
) -> tuple[str, str]:
    """Run the launcher through calibration without starting a model server."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calibration_log = tmp_path / "calibration-args.log"
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" > "${FAKE_CALIBRATION_ARGS_LOG}"
if [[ "${FAKE_CALIBRATION_EXIT}" != "0" ]]; then
  exit "${FAKE_CALIBRATION_EXIT}"
fi
printf '%s\\n' "${FAKE_CALIBRATION_POLICY}"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_vllm = bin_dir / "vllm"
    fake_vllm.write_text(
        """#!/usr/bin/env bash
printf 'fake-vllm policy=%s limit=%s\\n' \
  "${B12X_PCIE_PLAIN_ALLREDUCE_POLICY:-unset}" \
  "${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-unset}" >&2
""",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    policy = (
        '{"custom_rows":[1,2,4,8,16,24],"dtype":"bfloat16",'
        '"hidden_size":4096,"measured_rows":[1,2,4,8,16,24,32],'
        '"version":1,"world_size":4}'
    )
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "DRY_RUN": "0",
        "MODE": "mtp0",
        "TP": "4",
        "FAKE_CALIBRATION_ARGS_LOG": str(calibration_log),
        "FAKE_CALIBRATION_EXIT": str(calibration_exit),
        "FAKE_CALIBRATION_POLICY": policy,
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(_LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stderr, calibration_log.read_text(encoding="utf-8")


def test_ds4_launcher_auto_selects_calibrated_b12x_for_tp2(tmp_path: Path) -> None:
    """Verify that TP2 uses B12X with startup shape calibration."""
    output = _dry_run(tmp_path, TP="2")

    assert "allreduce=b12x" in output
    assert "plain_ar_calibration=skipped:dry-run" in output


def test_ds4_launcher_auto_keeps_b12x_for_tp4(tmp_path: Path) -> None:
    """Verify that automatic selection retains B12X at TP4."""
    output = _dry_run(tmp_path, TP="4")

    assert "allreduce=b12x" in output
    assert "plain_ar_calibration=skipped:dry-run" in output


def test_ds4_launcher_explicit_allreduce_overrides_auto(tmp_path: Path) -> None:
    """Verify that an explicit all-reduce mode overrides automatic selection."""
    output = _dry_run(tmp_path, TP="2", ALLREDUCE_MODE="b12x")

    assert "allreduce=b12x" in output


def test_ds4_launcher_preserves_explicit_flashinfer_ipc(tmp_path: Path) -> None:
    """Verify that FlashInfer IPC remains available as an explicit control."""
    output = _dry_run(tmp_path, TP="2", ALLREDUCE_MODE="flashinfer-ipc")

    assert "allreduce=flashinfer-ipc" in output
    assert "plain_ar_calibration=disabled" in output


def test_ds4_launcher_numeric_b12x_limit_disables_calibration(tmp_path: Path) -> None:
    """Verify that a numeric route limit remains an operator-owned policy."""
    output = _dry_run(
        tmp_path,
        TP="4",
        ALLREDUCE_MODE="b12x",
        VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="64KB",
    )

    assert "allreduce=b12x" in output
    assert "plain_ar_calibration=disabled" in output


def test_ds4_launcher_can_disable_b12x_calibration(tmp_path: Path) -> None:
    """Verify that calibration can be disabled without changing the backend."""
    output = _dry_run(
        tmp_path,
        TP="4",
        B12X_PCIE_PLAIN_ALLREDUCE_CALIBRATION="off",
    )

    assert "allreduce=b12x" in output
    assert "plain_ar_calibration=off" in output


def test_ds4_launcher_exports_successful_calibration_policy(tmp_path: Path) -> None:
    """Verify that the measured policy reaches the model-server process."""
    output, calibration_args = _run_with_fake_calibration(tmp_path)

    assert "plain_ar_calibration=ready" in output
    assert 'fake-vllm policy={"custom_rows":[1,2,4,8,16,24]' in output
    assert "limit=auto" in output
    assert "--world-size 4" in calibration_args
    assert "--rows 1,2,4,8,16,24,32" in calibration_args


def test_ds4_launcher_uses_conservative_limit_when_calibration_fails(
    tmp_path: Path,
) -> None:
    """Verify that a failed probe cannot extend B12X beyond 64 KiB."""
    output, _ = _run_with_fake_calibration(tmp_path, calibration_exit=4)

    assert "plain_ar_calibration=failed:64KB-fallback" in output
    assert "WARNING: B12X plain all-reduce calibration failed" in output
    assert "fake-vllm policy=unset limit=64KB" in output
