from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.agents.codex import deployer as deployer_module
from ale_run.agents.codex.config import CodexConfig
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

    monkeypatch.setattr(deployer_module, "_VENDOR_BINARY_WIN_TOPLEVEL", str(top_level))
    monkeypatch.setattr(deployer_module, "_VENDOR_BINARY_WIN_NESTED", str(nested))
    monkeypatch.setattr(deployer_module.subprocess, "run", fake_run)
    monkeypatch.setattr(deployer_module.time, "sleep", lambda _: None)

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


def test_qemu_stages_host_verified_binary_through_exchange_share(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"host-cached-codex"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    slot_root = tmp_path / "runtime" / "slots" / "run-1"
    exchange_dir = slot_root / "exchange"
    exchange_dir.mkdir(parents=True)
    config = CodexConfig(
        patched_binary_url_windows="https://example.test/codex.exe",
        patched_binary_sha256_windows=expected_sha256,
    )
    sandbox = SimpleNamespace(
        is_linux=False,
        metadata={
            "provider": "qemu",
            "slot_root": str(slot_root),
            "exchange_host_dir": str(exchange_dir),
            "exchange_guest_share": r"\\host.lan\Data",
        },
    )

    def fake_cache(_: str, digest: str, destination: Path) -> None:
        assert digest == expected_sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(deployer_module, "_ensure_cached_asset", fake_cache)

    asyncio.run(CodexDeployer.stage_sandbox_assets(config=config, sandbox=sandbox))

    staged_name = f"codex-{expected_sha256}.bin"
    assert (exchange_dir / staged_name).read_bytes() == payload
    assert config.patched_binary_staged_path == rf"\\host.lan\Data\{staged_name}"


def test_linux_qemu_keeps_guest_download_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CodexConfig()
    sandbox = SimpleNamespace(
        is_linux=True,
        metadata={"provider": "qemu"},
    )
    monkeypatch.setattr(
        deployer_module,
        "_ensure_cached_asset",
        lambda *_: pytest.fail("Linux QEMU should not stage an unmounted SMB path"),
    )

    asyncio.run(CodexDeployer.stage_sandbox_assets(config=config, sandbox=sandbox))

    assert config.patched_binary_staged_path == ""


def test_qemu_staging_rejects_unsafe_digest(tmp_path: Path) -> None:
    slot_root = tmp_path / "runtime" / "slots" / "run-1"
    exchange_dir = slot_root / "exchange"
    exchange_dir.mkdir(parents=True)
    config = CodexConfig(
        patched_binary_url_windows="https://example.test/codex.exe",
    )
    config.patched_binary_sha256_windows = "../unsafe"
    sandbox = SimpleNamespace(
        is_linux=False,
        metadata={
            "provider": "qemu",
            "slot_root": str(slot_root),
            "exchange_host_dir": str(exchange_dir),
            "exchange_guest_share": r"\\host.lan\Data",
        },
    )

    with pytest.raises(RuntimeError, match="64 hexadecimal"):
        asyncio.run(CodexDeployer.stage_sandbox_assets(config=config, sandbox=sandbox))


def test_staged_binary_avoids_guest_network_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"host-staged-codex"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "shared-codex.exe"
    target = tmp_path / "vendor-codex.exe"
    source.write_bytes(payload)
    target.write_bytes(b"stale")

    monkeypatch.setattr(deployer_module, "_VENDOR_BINARY_WIN_TOPLEVEL", str(target))
    monkeypatch.setattr(
        deployer_module,
        "_VENDOR_BINARY_WIN_NESTED",
        str(tmp_path / "missing.exe"),
    )
    monkeypatch.setattr(
        deployer_module,
        "_download_pinned_asset",
        lambda *_: pytest.fail("guest network download should not run"),
    )

    asyncio.run(
        _deployer_for_windows()._replace_native_binary(
            "https://example.test/codex.exe",
            expected_sha256,
            str(source),
        )
    )

    assert target.read_bytes() == payload
