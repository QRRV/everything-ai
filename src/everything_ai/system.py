# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""A quick look at the learner's computer, so lessons can say upfront what works on it.

No PyTorch here: this runs at every start and must be fast. Lessons that train models
check the device themselves.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _nvidia() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, mem = (part.strip() for part in out[0].split(",", 1))
    try:
        vram = round(float(mem) / 1024)
    except ValueError:
        vram = None
    return {"kind": "cuda", "name": name, "vram_gb": vram}


def _is_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def probe() -> dict:
    system = platform.system()
    if system == "Darwin":
        os_name = "macOS"
    elif _is_wsl():
        os_name = "Linux (WSL2)"
    else:
        os_name = system
    gpu = _nvidia()
    if gpu is None and system == "Darwin" and platform.machine() == "arm64":
        gpu = {"kind": "mps", "name": "Apple silicon", "vram_gb": None}
    ram = _ram_gb()
    return {
        "os": os_name,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "ram_gb": round(ram) if ram else None,
        "gpu": gpu,
    }
