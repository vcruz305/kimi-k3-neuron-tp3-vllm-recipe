#!/usr/bin/env python3
"""Hermetic release audit for the public patch recipe."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {"", ".md", ".py", ".sh", ".json", ".env", ".yml", ".yaml", ".txt", ".patch"}
FORBIDDEN_TEXT = (
    "release_eval",
    "dspark_tp3",
    "0010-dflash",
    "/artifacts",
    "/workspace/",
)
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "API secret": re.compile(r"sk_[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

EXPECTED_PLUGIN_PATCHES = {
    "0001-feat-add-Kimi-K3-Neuron-GGUF-adapter.patch",
    "0005-fix-kimi-k3-dequantize-latent-projections-for-native.patch",
    "0007-fix-moe-guard-CUDA-vector-grid-z-limit.patch",
    "0009-fix-honor-distributed-TP-rank-in-GGUF-fused-loaders.patch",
}
EXPECTED_VLLM_PATCHES = {
    "0004-fix-kimi-k3-pad-vocabulary-for-odd-TP-sizes.patch",
    "0008-fix-kimi-k3-preserve-precision-sensitive-GGUF-weight.patch",
    "0010-kimi-k3-dspark-gguf-target-tp3.patch",
    "0012-dspark-draft-config-format-isolation.patch",
    "0013-kimi-linear-eagle3-target-bridge.patch",
    "0014-dflash-full-cg-with-piecewise-target.patch",
}
EXPECTED_OPTIONAL_PATCHES = {"0011-hopper-flashmla-noncausal-dspark.patch"}


def text_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in TEXT_SUFFIXES
    ]


def check_public_hygiene() -> None:
    errors: list[str] = []
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT)
        if relative.as_posix() != "scripts/check_bundle.py":
            for forbidden in FORBIDDEN_TEXT:
                if forbidden in text:
                    errors.append(
                        f"{relative}: forbidden private/stale text {forbidden!r}"
                    )
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{relative}: possible {label}")
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".gguf", ".safetensors", ".pem", ".key"}:
            errors.append(f"forbidden artifact: {path.relative_to(ROOT)}")
    if errors:
        raise AssertionError("\n".join(errors))


def names(directory: str) -> set[str]:
    return {path.name for path in (ROOT / directory).glob("*.patch")}


def check_patch_set() -> None:
    assert names("patches/gguf-plugin") == EXPECTED_PLUGIN_PATCHES
    assert names("patches/vllm") == EXPECTED_VLLM_PATCHES
    assert names("patches/optional") == EXPECTED_OPTIONAL_PATCHES
    all_names = EXPECTED_PLUGIN_PATCHES | EXPECTED_VLLM_PATCHES
    assert not any(name.startswith(("0002-", "0003-", "0006-")) for name in all_names)


def check_hash_allowlist() -> None:
    lines = [
        line
        for line in (ROOT / "SHA256SUMS").read_text().splitlines()
        if line.strip()
    ]
    assert lines, "SHA256SUMS is empty"
    seen: set[str] = set()
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = ROOT / relative
        assert path.is_file(), f"missing allowlisted file: {relative}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest.lower(), f"hash mismatch: {relative}"
        assert relative not in seen, f"duplicate hash entry: {relative}"
        seen.add(relative)

    expected = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for directory in ("patches/gguf-plugin", "patches/vllm", "patches/optional")
        for path in (ROOT / directory).glob("*.patch")
    }
    expected |= {"config/config.json", "pins.env"}
    assert seen == expected, f"SHA256SUMS coverage differs: {seen ^ expected}"


def check_config_and_pins() -> None:
    config = json.loads((ROOT / "config/config.json").read_text())
    expected = {
        "model_type": "kimi_linear",
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "moe_intermediate_size": 1536,
        "num_shared_experts": 4,
        "vocab_size": 163840,
    }
    for key, value in expected.items():
        assert config.get(key) == value, f"config {key} changed"
    assert config["architectures"] == ["KimiLinearForCausalLM"]

    pins = dict(
        line.split("=", 1)
        for line in (ROOT / "pins.env").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert pins["VLLM_COMMIT"] == "75231eff2f3873e2bce7cc9558bb5227ea70b808"
    assert pins["GGUF_PLUGIN_COMMIT"] == "d94067060884ea87766f12010c3a8b9c2d6715cc"
    assert pins["GGUF_PYTHON_VERSION"] == "0.19.0"
    assert pins["TARGET_MODEL_REVISION"] == "fc23910006796671aecd5551d425b5e77b61d2f2"
    assert pins["TARGET_TOKENIZER_REVISION"] == "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    assert pins["DSPARK_MODEL_REVISION"] == "cf6b8244620e7ea4b0651d214f28e89eac75bed6"


def check_local_markdown_links() -> None:
    pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    errors: list[str] = []
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target = target.strip().strip("<>")
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean_target = target.split("#", 1)[0]
            resolved = (path.parent / clean_target).resolve()
            if ROOT not in resolved.parents and resolved != ROOT:
                errors.append(f"{path.relative_to(ROOT)}: link escapes repository: {target}")
            elif not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)}: broken link: {target}")
    if errors:
        raise AssertionError("\n".join(errors))


def main() -> None:
    check_public_hygiene()
    check_patch_set()
    check_hash_allowlist()
    check_config_and_pins()
    check_local_markdown_links()
    print("PASS: hashes, patch set, config, links, and public hygiene")


if __name__ == "__main__":
    main()
