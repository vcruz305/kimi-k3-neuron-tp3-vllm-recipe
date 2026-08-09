#!/usr/bin/env python3
"""No-hardware-damage GPU architecture preflight for this recipe.

Run this FIRST, before scripts/prepare_sources.sh or
scripts/build_from_source.sh. It only queries the driver (via torch) and
introspects installed Python packages; it never allocates GPU memory,
launches a kernel, writes a file, or touches a token or model file.

What it does:

- reports torch.cuda.get_device_capability() per visible GPU and
  torch.cuda.get_arch_list();
- decides whether the installed torch build actually carries kernels for
  the detected capability -- a native cubin, a PTX entry that will
  JIT-compile forward onto it, or neither;
- enumerates the four MLA attention backends this recipe cares about
  (FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA) by importing each
  one through vLLM's own AttentionBackendEnum registry and calling its
  supports_compute_capability() classmethod, rather than assuming from a
  hardcoded table which archs each backend supports;
- prints GO / GO-WITH-CAVEATS / NO-GO plus a recommended attention_backend
  and environment variables for the detected architecture.

Exit codes:
  0  GO or GO-WITH-CAVEATS (see the printed caveats before spending GPU time)
  1  NO-GO
  2  usage error (argparse)

Only sm_90 (Hopper, e.g. H100/H200) is GPU-validated by this recipe -- see
evidence/DSPARK-TP3-H200.md. Every other architecture this script has an
opinion about is UNTESTED; "GO-WITH-CAVEATS" means "plausible from the
installed kernels and vLLM's own capability gates," not "measured."

--json emits a machine-readable report intended to be safe to paste into a
bug report: no tokens, no absolute filesystem paths, no model files. Error
text that might contain a path is redacted before it is included.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

MLA_BACKENDS = ("FLASH_ATTN_MLA", "FLASHMLA", "FLASHINFER_MLA", "TRITON_MLA")

# ---------------------------------------------------------------------------
# Curated, recipe-specific knowledge. Kept separate from the detected facts
# below so it is obvious in both the JSON and the human report which parts
# are measured/introspected right now and which parts are prior knowledge
# written down by this recipe's authors (and can go stale).
#
# Keyed by (major, minor) compute capability, i.e. the exact tuple
# torch.cuda.get_device_capability() returns.
# ---------------------------------------------------------------------------
KNOWN_ARCHES: dict[tuple[int, int], dict[str, Any]] = {
    (9, 0): {
        "name": "Hopper, e.g. H100/H200",
        "status": "VALIDATED",
        "notes": [
            "The only architecture with a GPU-measured result from this "
            "recipe: 42.464 tok/s prose (num_speculative_tokens=2), "
            "52.454 tok/s coding (num_speculative_tokens=3), against a "
            "34.875 tok/s contemporaneous target-only baseline. See "
            "evidence/DSPARK-TP3-H200.md.",
            "patches/optional/0011-hopper-flashmla-noncausal-dspark.patch "
            "applies here, and only here (set "
            "K3_APPLY_OPTIONAL_HOPPER_FLASHMLA=1 before "
            "scripts/prepare_sources.sh to include it).",
        ],
        "recommended_backend": "TRITON_MLA",
        "recommended_backend_notes": (
            "TRITON_MLA is what docs/INTEGRATED-RUNBOOK.md uses for the "
            "first correctness run. FLASH_ATTN_MLA and FLASHMLA also "
            "self-report support for sm_90 in this vLLM pin (see the "
            "detected backends below) and are the ones to compare once "
            "TRITON_MLA is qualified."
        ),
        "recommended_env": {},
        "references": [],
    },
    (12, 0): {
        "name": "Blackwell workstation, e.g. RTX PRO 6000",
        "status": "UNTESTED, known risk",
        "notes": [
            "Open upstream issue: vLLM does not support DeepSeek-series "
            "models on RTX PRO 6000 / SM120 -- "
            "https://github.com/vllm-project/vllm/issues/26211. Kimi-K3 "
            "is DeepSeek-lineage MLA, so this plausibly applies here too.",
            "FLASHMLA's own supports_compute_capability() in this vLLM "
            "pin accepts capability.major in [9, 10] only -- sm_120 "
            "(major=12) is not in that set, so FLASHMLA is not expected "
            "to be offered here regardless of import success.",
        ],
        "recommended_backend": "TRITON_MLA",
        "recommended_backend_notes": (
            "TRITON_MLA's supports_compute_capability() unconditionally "
            "returns True in this vLLM pin, which makes it the "
            "arch-agnostic fallback. Do not assume the plain "
            "FLASHINFER_MLA backend works here: in this vLLM pin its "
            "supports_compute_capability() accepts capability.major == 10 "
            "only (Blackwell *datacenter*, e.g. B200/GB200), not major "
            "== 12 -- check the 'detected backends' section below against "
            "this actual install rather than trusting this note. A "
            "separate FLASHINFER_MLA_SPARSE_SM120 backend exists in the "
            "same vLLM tree and targets sm_120 by name, but it belongs to "
            "the sparse-MLA family (a different metadata/impl contract "
            "for indexer-based models) and this recipe's patches do not "
            "wire Kimi-K3 into that path -- treat it as a pointer for "
            "future work, not a validated option."
        ),
        "recommended_env": {"VLLM_FLASH_ATTN_VERSION": "2"},
        "references": ["https://github.com/vllm-project/vllm/issues/26211"],
    },
    (12, 1): {
        "name": "NVIDIA DGX Spark, GB10, aarch64",
        "status": "UNTESTED, known risk",
        "notes": [
            "Open upstream issue: no sm_121 (Blackwell) support on "
            "aarch64 -- https://github.com/vllm-project/vllm/issues/36821.",
            "Stock PyTorch has shipped native kernels only through sm_120 "
            "(confirmed by inspecting torch.cuda.get_arch_list() output "
            "from an official PyTorch wheel while writing this script -- "
            "it had no sm_121 and no compute_120 PTX entry at all). "
            "sm_121 needs either a build with PTX coverage at or below "
            "12.1 (see scripts/build_from_source.sh's TORCH_CUDA_ARCH_LIST) "
            "or a genuine from-source torch/vLLM/GGUF-plugin build. Check "
            "the 'kernel coverage' section below for what THIS install "
            "actually has, not what stock wheels have.",
            "Memory: docs/DGX-SPARK-PORT.md measured 84.06 GiB/rank at "
            "TP4 for the 307.49 GiB model -- 4 Sparks, not 3. A larger, "
            "in-progress ~353.71 GiB IQ2-class build would scale that to "
            "roughly 88 GiB/rank at TP4 and would not fit 3 Sparks either.",
        ],
        "recommended_backend": "TRITON_MLA",
        "recommended_backend_notes": (
            "Same reasoning as sm_120 (same Blackwell major SM family, "
            "major=12): TRITON_MLA is the arch-agnostic fallback. Neither "
            "FLASHMLA (major in [9, 10]) nor the plain FLASHINFER_MLA "
            "backend (major == 10) self-report support for major == 12 "
            "in this vLLM pin."
        ),
        "recommended_env": {"VLLM_FLASH_ATTN_VERSION": "2"},
        "references": ["https://github.com/vllm-project/vllm/issues/36821"],
    },
    (8, 0): {
        "name": "Ampere datacenter, e.g. A100",
        "status": "UNTESTED",
        "notes": [
            "Not GPU-tested by this recipe. This gap is not "
            "Blackwell-specific -- sm_80 is exactly as unvalidated as "
            "sm_120 or sm_121, it has just attracted less attention "
            "because it raises no known upstream blocker.",
        ],
        "recommended_backend": "TRITON_MLA",
        "recommended_backend_notes": (
            "FLASH_ATTN_MLA (major == 9) and FLASHMLA (major in [9, 10]) "
            "are not expected on sm_80 per their own capability gates in "
            "this vLLM pin. TRITON_MLA is the one to try first."
        ),
        "recommended_env": {},
        "references": [],
    },
}

_WIN_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")
_POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:[\w.\-]+/)+[\w.\-]*")


def _sanitize(text: str | None) -> str | None:
    """Redact substrings that look like absolute filesystem paths.

    Exception text (ImportError, OSError, ...) can embed the path to a
    .so/.pyd/site-packages file, which can carry a username on Windows.
    This report is meant to be safe to paste into a public bug report, so
    redact anything path-shaped rather than trust every call site.
    """
    if not text:
        return text
    text = _WIN_PATH_RE.sub("<path-redacted>", text)
    text = _POSIX_PATH_RE.sub("<path-redacted>", text)
    return text


def _err(exc: BaseException) -> str:
    return _sanitize(str(exc)) or repr(exc)


def read_pins(repo_root: Path) -> dict[str, str]:
    """Best-effort, informational-only read of pins.env. Never raises."""
    pins_path = repo_root / "pins.env"
    pins: dict[str, str] = {}
    try:
        for line in pins_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            pins[key.strip()] = value.strip()
    except OSError:
        pass
    return pins


def _normalize_sm_token(token: str) -> str | None:
    """"sm_90" -> "sm_90"; "sm_90a" -> "sm_90"; "sm_120f" -> "sm_120"."""
    if not token.startswith("sm_"):
        return None
    return "sm_" + token[len("sm_") :].rstrip("af")


def _parse_compute_token(token: str) -> tuple[int, int] | None:
    """"compute_90" -> (9, 0); "compute_120a" -> (12, 0). None if unparseable."""
    if not token.startswith("compute_"):
        return None
    digits = token[len("compute_") :].rstrip("af")
    if len(digits) < 2 or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


def classify_kernel_coverage(
    capability: tuple[int, int], arch_list: list[str]
) -> tuple[str, str]:
    """Return (coverage, detail) where coverage is native/ptx_jit/none.

    native  : arch_list has a cubin built for exactly this capability.
    ptx_jit : no native cubin, but arch_list has a compute_XY PTX entry
              with XY <= capability. CUDA guarantees PTX compiled for one
              capability JIT-compiles forward onto any device of equal or
              greater capability (CUDA C++ Programming Guide, "PTX
              Compatibility"), so this will work but pays a JIT-compile
              pause the first time each kernel runs.
    none    : neither. General CUDA kernels are not expected to run.
    """
    sm_token = f"sm_{capability[0]}{capability[1]}"
    normalized = {_normalize_sm_token(tok) for tok in arch_list}
    if sm_token in normalized:
        return "native", f"{sm_token} cubin present in torch.cuda.get_arch_list()"

    candidates = []
    for tok in arch_list:
        parsed = _parse_compute_token(tok)
        if parsed is not None and parsed <= capability:
            candidates.append((parsed, tok))
    if candidates:
        candidates.sort()
        (major, minor), tok = candidates[-1]
        same_family = major == capability[0]
        family_note = (
            "same major SM family"
            if same_family
            else "cross-family JIT -- larger generational gap, less tested in practice"
        )
        return "ptx_jit", (
            f"no native {sm_token} cubin, but {tok} PTX is present and will "
            f"JIT-compile on first kernel load ({family_note})"
        )
    return "none", (
        f"no native {sm_token} cubin and no compute_* PTX at or below "
        f"{capability[0]}.{capability[1]} in torch.cuda.get_arch_list()"
    )


def detect_torch_and_gpus() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return {"torch_installed": False, "error": _err(exc)}

    info: dict[str, Any] = {
        "torch_installed": True,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torch_cuda_build_version": getattr(torch.version, "cuda", None),
        "cuda_available": False,
        "arch_list": [],
        "gpus": [],
    }

    try:
        info["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception as exc:  # noqa: BLE001
        info["arch_list_error"] = _err(exc)

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        info["cuda_available_error"] = _err(exc)
        return info

    if not info["cuda_available"]:
        return info

    try:
        count = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        info["device_count_error"] = _err(exc)
        return info

    for i in range(count):
        gpu: dict[str, Any] = {"index": i}
        try:
            gpu["name"] = torch.cuda.get_device_name(i)
        except Exception as exc:  # noqa: BLE001
            gpu["name_error"] = _err(exc)
        try:
            major, minor = torch.cuda.get_device_capability(i)
            gpu["capability"] = [major, minor]
            gpu["sm"] = f"sm_{major}{minor}"
        except Exception as exc:  # noqa: BLE001
            gpu["capability_error"] = _err(exc)
        info["gpus"].append(gpu)

    return info


def probe_mla_backends(capability: tuple[int, int]) -> dict[str, dict[str, Any]]:
    """Import each MLA backend through vLLM's own registry and ask it,
    rather than hardcoding which archs each backend supports."""
    results: dict[str, dict[str, Any]] = {}

    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception as exc:  # noqa: BLE001
        reason = f"vllm.v1.attention.backends.registry unavailable: {_err(exc)}"
        for name in MLA_BACKENDS:
            results[name] = {
                "importable": False,
                "import_error": reason,
                "supports_this_capability": None,
            }
        return results

    cap_obj = None
    try:
        from vllm.platforms.interface import DeviceCapability

        cap_obj = DeviceCapability(major=capability[0], minor=capability[1])
    except Exception:  # noqa: BLE001
        cap_obj = None  # fall through; per-backend support stays unknown

    for name in MLA_BACKENDS:
        entry: dict[str, Any] = {
            "importable": False,
            "import_error": None,
            "supports_this_capability": None,
        }
        try:
            member = AttentionBackendEnum[name]
            cls = member.get_class()
            entry["importable"] = True
            entry["class_path"] = member.get_path()
        except Exception as exc:  # noqa: BLE001
            entry["import_error"] = _err(exc)
            results[name] = entry
            continue

        if cap_obj is not None:
            try:
                entry["supports_this_capability"] = bool(
                    cls.supports_compute_capability(cap_obj)
                )
            except Exception as exc:  # noqa: BLE001
                entry["support_check_error"] = _err(exc)
        results[name] = entry

    return results


def build_report() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    report: dict[str, Any] = {
        "recipe": "kimi-k3-neuron-tp3-vllm-recipe",
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "recipe_pins": {
            key: value
            for key, value in read_pins(repo_root).items()
            if key
            in (
                "VLLM_COMMIT",
                "GGUF_PLUGIN_COMMIT",
                "SOURCE_BUILD_CUDA_MAJOR_MINOR",
                "PYTHON_MAJOR_MINOR",
            )
        },
    }

    torch_info = detect_torch_and_gpus()
    report["torch"] = torch_info
    report["gpus"] = []

    if not torch_info.get("torch_installed"):
        report["verdict"] = "NO-GO"
        report["verdict_reasons"] = [
            "torch is not importable in this Python environment. Install "
            "this recipe's pinned build/runtime requirements first (see "
            "APPLY.md), then re-run this script.",
        ]
        return report

    if not torch_info.get("cuda_available"):
        reason = torch_info.get("cuda_available_error") or (
            "torch.cuda.is_available() is False -- this torch build has no "
            "usable CUDA (a CPU-only wheel, a missing/mismatched driver, or "
            "no GPU visible to this process)."
        )
        report["verdict"] = "NO-GO"
        report["verdict_reasons"] = [reason]
        return report

    gpus = torch_info["gpus"]
    if not gpus:
        report["verdict"] = "NO-GO"
        report["verdict_reasons"] = [
            "torch.cuda.is_available() is True but device_count() found no GPUs.",
        ]
        return report

    arch_list = torch_info.get("arch_list", [])
    per_gpu = []
    capabilities: set[tuple[int, int]] = set()
    for gpu in gpus:
        entry = {"index": gpu["index"], "name": gpu.get("name", "unknown"), "sm": gpu.get("sm")}
        cap = gpu.get("capability")
        if cap is None:
            entry["kernel_coverage"] = "unknown"
            entry["kernel_coverage_detail"] = gpu.get(
                "capability_error", "capability query failed"
            )
        else:
            capabilities.add(tuple(cap))
            coverage, detail = classify_kernel_coverage(tuple(cap), arch_list)
            entry["kernel_coverage"] = coverage
            entry["kernel_coverage_detail"] = detail
        per_gpu.append(entry)
    report["gpus"] = per_gpu

    unknown_gpus = [g for g in per_gpu if g["kernel_coverage"] == "unknown"]
    if unknown_gpus:
        report["verdict"] = "NO-GO"
        report["verdict_reasons"] = [
            f"failed to query compute capability for {len(unknown_gpus)} of "
            f"{len(per_gpu)} visible GPU(s); see gpus[].kernel_coverage_detail.",
        ]
        return report

    if len(capabilities) > 1:
        report["verdict"] = "NO-GO"
        report["verdict_reasons"] = [
            f"visible GPUs report different compute capabilities "
            f"({sorted(capabilities)}). This recipe assumes a homogeneous "
            f"GPU set (see pins.env / README hardware assumptions).",
        ]
        return report

    capability = next(iter(capabilities))
    coverage = per_gpu[0]["kernel_coverage"]
    sm_label = f"sm_{capability[0]}{capability[1]}"
    report["detected_capability"] = list(capability)
    report["detected_sm"] = sm_label

    backends = probe_mla_backends(capability)
    report["mla_backends"] = backends
    working_backends = [
        name
        for name in MLA_BACKENDS
        if backends.get(name, {}).get("importable")
        and backends[name].get("supports_this_capability")
    ]
    report["working_backends"] = working_backends

    known = KNOWN_ARCHES.get(capability)
    reasons: list[str] = []

    if coverage == "none":
        verdict = "NO-GO"
        reasons.append(
            f"installed torch has neither a native cubin nor a PTX entry "
            f"covering {sm_label} ({per_gpu[0]['kernel_coverage_detail']}). "
            f"General CUDA kernels are not expected to run."
        )
    elif not working_backends:
        verdict = "NO-GO"
        reasons.append(
            f"torch has {coverage} kernel coverage for {sm_label}, but no "
            f"MLA attention backend in this vLLM install both imports "
            f"cleanly and self-reports support for it. This usually means "
            f"vLLM is not installed yet, or TRITON_MLA (the arch-agnostic "
            f"fallback) failed to import -- see "
            f"mla_backends.TRITON_MLA.import_error above."
        )
    elif known is not None and known["status"] == "VALIDATED" and coverage == "native":
        verdict = "GO"
        reasons.append(f"{sm_label} is this recipe's GPU-validated architecture.")
    else:
        verdict = "GO-WITH-CAVEATS"
        if known is not None and known["status"] == "VALIDATED":
            reasons.append(
                f"{sm_label} ({known['name']}) is this recipe's validated "
                f"architecture, but kernel coverage here is '{coverage}', "
                f"not the native-cubin case the measured numbers were "
                f"produced under -- treat this specific install as unverified."
            )
        elif known is not None:
            reasons.append(
                f"{sm_label} ({known['name']}) is {known['status']} by this "
                f"recipe. Kernel coverage is '{coverage}' and at least one "
                f"MLA backend ({', '.join(working_backends)}) claims "
                f"support, so it is plausible but not proven."
            )
        else:
            reasons.append(
                f"{sm_label} is not in this recipe's known compatibility "
                f"matrix at all (see README.md). Kernel coverage is "
                f"'{coverage}' and {', '.join(working_backends)} claim "
                f"support, so it may work, but nobody has tried it with "
                f"this recipe."
            )
        if coverage == "ptx_jit":
            reasons.append(
                "kernel coverage is via PTX JIT, not a native cubin -- "
                "expect a pause while the driver compiles the first kernel "
                "launch. This check also only covers torch's own kernels: "
                "this recipe's hand-written GGUF-plugin CUDA kernels (MMVQ, "
                "MoE dispatch) are compiled separately during the source "
                "build and are not covered by this preflight check at all."
            )
        if capability[0] == 12 and platform.machine().lower() in ("aarch64", "arm64"):
            reasons.append(
                "aarch64 + Blackwell (sm_12x): confirm patch 0007's CUDA "
                "grid-limit guard has been re-validated on this capability "
                "-- docs/DGX-SPARK-PORT.md flags it as unchecked."
            )

    report["verdict"] = verdict
    report["verdict_reasons"] = reasons

    if known is not None:
        report["known_arch_notes"] = known["notes"]
        report["recommended_attention_backend"] = known["recommended_backend"]
        report["recommended_attention_backend_notes"] = known["recommended_backend_notes"]
        report["recommended_env"] = known["recommended_env"]
        report["references"] = known["references"]
    elif working_backends:
        report["recommended_attention_backend"] = (
            "TRITON_MLA" if "TRITON_MLA" in working_backends else working_backends[0]
        )
        report["recommended_attention_backend_notes"] = (
            "No curated guidance for this architecture in this script; this "
            "recommendation is derived only from which MLA backends "
            "self-report support just now."
        )
        report["recommended_env"] = {}
        report["references"] = []

    return report


def format_human(report: dict[str, Any]) -> str:
    lines = []
    rule = "=" * 78
    lines.append(rule)
    lines.append("k3-neuron-tp3-vllm-recipe: architecture preflight")
    lines.append(rule)
    lines.append(
        f"python: {report['python_version']}  "
        f"platform: {report['platform_system']}/{report['platform_machine']}"
    )
    pins = report.get("recipe_pins") or {}
    if pins:
        lines.append(
            "recipe pins: vLLM " + pins.get("VLLM_COMMIT", "?")[:12]
            + ", GGUF plugin " + pins.get("GGUF_PLUGIN_COMMIT", "?")[:12]
            + ", CUDA " + pins.get("SOURCE_BUILD_CUDA_MAJOR_MINOR", "?")
        )

    torch_info = report["torch"]
    if not torch_info.get("torch_installed"):
        lines.append(f"torch: NOT INSTALLED ({torch_info.get('error')})")
    else:
        lines.append(
            f"torch: {torch_info.get('torch_version')} "
            f"(cuda build {torch_info.get('torch_cuda_build_version')})"
        )
        lines.append(f"torch.cuda.get_arch_list(): {torch_info.get('arch_list')}")
        lines.append(f"torch.cuda.is_available(): {torch_info.get('cuda_available')}")

    lines.append("")
    lines.append("GPUs:")
    gpus = report.get("gpus") or []
    if not gpus:
        lines.append("  (none detected)")
    for gpu in gpus:
        if gpu.get("kernel_coverage") == "unknown":
            lines.append(
                f"  [{gpu['index']}] {gpu.get('name', 'unknown')}: "
                f"capability query failed ({gpu.get('kernel_coverage_detail')})"
            )
        else:
            lines.append(
                f"  [{gpu['index']}] {gpu.get('name')}  {gpu.get('sm')}  "
                f"kernel_coverage={gpu.get('kernel_coverage')} "
                f"({gpu.get('kernel_coverage_detail')})"
            )

    if "mla_backends" in report:
        lines.append("")
        lines.append("MLA attention backends in this install:")
        for name in MLA_BACKENDS:
            info = report["mla_backends"].get(name, {})
            if not info.get("importable"):
                lines.append(f"  {name:16s} NOT IMPORTABLE  ({info.get('import_error')})")
            else:
                support = info.get("supports_this_capability")
                support_str = (
                    "SUPPORTS" if support else ("REFUSES" if support is False else "UNKNOWN")
                )
                lines.append(
                    f"  {name:16s} importable, self-reports: {support_str} this capability"
                )

    lines.append("")
    lines.append(f"VERDICT: {report['verdict']}")
    for reason in report.get("verdict_reasons", []):
        lines.append(f"  - {reason}")

    if report.get("known_arch_notes"):
        lines.append("")
        lines.append("Known notes for this architecture:")
        for note in report["known_arch_notes"]:
            lines.append(f"  - {note}")

    if report.get("recommended_attention_backend"):
        lines.append("")
        lines.append(f"Recommended attention_backend: {report['recommended_attention_backend']}")
        if report.get("recommended_attention_backend_notes"):
            lines.append(f"  {report['recommended_attention_backend_notes']}")
        if report.get("recommended_env"):
            lines.append("Recommended environment variables:")
            for key, value in report["recommended_env"].items():
                lines.append(f"  export {key}={value}")
        if report.get("references"):
            lines.append("References:")
            for ref in report["references"]:
                lines.append(f"  {ref}")

    lines.append(rule)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="No-hardware-damage GPU architecture preflight for this recipe."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON report instead of the human-readable one",
    )
    args = parser.parse_args()

    report = build_report()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_human(report))

    return 0 if report["verdict"] in ("GO", "GO-WITH-CAVEATS") else 1


if __name__ == "__main__":
    sys.exit(main())
