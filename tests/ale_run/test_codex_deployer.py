from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.agents.codex import deployer as deployer_module
from ale_run.agents.codex.deployer import CodexDeployer


def _deployer_for_windows() -> CodexDeployer:
    deployer = object.__new__(CodexDeployer)
    deployer._is_windows = True
    return deployer


def test_pinned_binary_download_retries_and_verifies_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"pinned-codex-binary"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    top_level = tmp_path / "top-level.exe"
    nested = tmp_path / "nested.exe"
    top_level.write_bytes(b"stale")
    nested.write_bytes(b"stale")
    attempts = 0

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        staged = Path(command[command.index("-o") + 1])
        if attempts < 3:
            return SimpleNamespace(returncode=28, stdout="", stderr="timeout")
        staged.write_bytes(payload)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(deployer_module, "_VENDOR_BINARY_WIN_TOPLEVEL", str(top_level))
    monkeypatch.setattr(deployer_module, "_VENDOR_BINARY_WIN_NESTED", str(nested))
    monkeypatch.setattr(deployer_module.subprocess, "run", fake_run)
    monkeypatch.setattr(deployer_module.asyncio, "sleep", no_sleep)

    asyncio.run(
        _deployer_for_windows()._replace_native_binary(
            "https://example.test/codex.exe",
            expected_sha256,
        )
    )

    assert attempts == 3
    assert top_level.read_bytes() == payload
    assert nested.read_bytes() == payload


def test_pinned_binary_digest_mismatch_does_not_replace_vendor_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    top_level = tmp_path / "top-level.exe"
    top_level.write_bytes(b"stale")

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        staged = Path(command[command.index("-o") + 1])
        staged.write_bytes(b"unexpected")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(deployer_module, "_VENDOR_BINARY_WIN_TOPLEVEL", str(top_level))
    monkeypatch.setattr(
        deployer_module,
        "_VENDOR_BINARY_WIN_NESTED",
        str(tmp_path / "missing.exe"),
    )
    monkeypatch.setattr(deployer_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        asyncio.run(
            _deployer_for_windows()._replace_native_binary(
                "https://example.test/codex.exe",
                "0" * 64,
            )
        )

    assert top_level.read_bytes() == b"stale"
