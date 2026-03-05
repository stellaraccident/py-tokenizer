#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build manylinux wheels for iree-tokenizer.

This script self-trampolines into a manylinux Docker/Podman container,
builds wheels for each requested Python version, and runs auditwheel
repair to produce portable manylinux wheels.

Usage (host):
    python build_tools/build_linux_packages.py
    python build_tools/build_linux_packages.py --docker podman
    python build_tools/build_linux_packages.py --python-versions cp312-cp312

Usage (CI — already inside the container):
    python build_tools/build_linux_packages.py --in-docker
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default manylinux images (same as IREE).
MANYLINUX_IMAGES = {
    "x86_64": "quay.io/pypa/manylinux_2_28_x86_64",
    "aarch64": "quay.io/pypa/manylinux_2_28_aarch64",
}

DEFAULT_PYTHON_VERSIONS = [
    "cp310-cp310",
    "cp311-cp311",
    "cp312-cp312",  # Produces abi3 wheel covering 3.12+
]


def get_iree_version() -> str:
    version_json = REPO_ROOT / "version.json"
    with open(version_json) as f:
        return json.load(f)["iree-version"]


def run(cmd: list[str], *, env: dict | None = None, check: bool = True, **kwargs):
    """Run a command, printing it first."""
    print(f"+ {' '.join(cmd)}", flush=True)
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, env=merged_env, check=check, **kwargs)


# ---------------------------------------------------------------------------
# Host mode: launch container
# ---------------------------------------------------------------------------


def run_on_host(args: argparse.Namespace):
    arch = platform.machine()
    image = args.image or MANYLINUX_IMAGES.get(arch)
    if not image:
        sys.exit(f"No default manylinux image for architecture {arch}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the command to re-invoke ourselves inside the container.
    inner_cmd = [
        "python3",
        f"{REPO_ROOT}/build_tools/build_linux_packages.py",
        "--in-docker",
        "--output-dir",
        str(output_dir),
        "--python-versions",
        *args.python_versions,
    ]
    if args.no_repair:
        inner_cmd.append("--no-repair")

    # Use :Z volume label for SELinux relabeling (needed on Fedora/RHEL
    # with rootless podman). Harmless on non-SELinux systems.
    vol_suffix = ":Z" if args.docker == "podman" else ""
    container_cmd = [
        args.docker,
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}{vol_suffix}",
        "-v",
        f"{output_dir}:{output_dir}{vol_suffix}",
        "-w",
        str(REPO_ROOT),
        image,
        "--",
        *inner_cmd,
    ]

    print(f"Launching {args.docker} with image {image}")
    run(container_cmd)

    print(f"\nWheels written to {output_dir}:")
    for whl in sorted(output_dir.glob("*.whl")):
        print(f"  {whl.name}")


# ---------------------------------------------------------------------------
# Docker mode: build wheels inside container
# ---------------------------------------------------------------------------


def clone_iree(tag: str, dest: Path):
    """Shallow-clone IREE sources if not already present."""
    if dest.exists():
        print(f"IREE sources already present at {dest}")
        return
    run(["git", "init", str(dest)])
    run(
        [
            "git",
            "-C",
            str(dest),
            "remote",
            "add",
            "origin",
            "https://github.com/iree-org/iree.git",
        ]
    )
    run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", tag])
    run(["git", "-C", str(dest), "checkout", "FETCH_HEAD"])
    # Init required submodules (flatcc for the tokenizer JSON parser,
    # benchmark because IREE's CMakeLists.txt unconditionally adds it).
    run(
        [
            "git",
            "-C",
            str(dest),
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "third_party/flatcc",
            "third_party/benchmark",
        ]
    )


def build_wheel(python: Path, output_dir: Path, iree_source_dir: Path):
    """Build one wheel for a given Python interpreter."""
    run(
        [
            str(python),
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "-v",
            "-w",
            str(output_dir),
            str(REPO_ROOT),
        ],
        env={"IREE_SOURCE_DIR": str(iree_source_dir)},
    )


def audit_wheel(wheel: Path, output_dir: Path):
    """Run auditwheel repair on a single wheel."""
    run(["auditwheel", "repair", "-w", str(output_dir), str(wheel)])
    wheel.unlink()
    print(f"  Removed unrepaired wheel: {wheel.name}")


def run_in_docker(args: argparse.Namespace):
    # Mark repo safe for git operations inside container.
    run(["git", "config", "--global", "--add", "safe.directory", "*"], check=False)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone IREE once for all Python versions.
    iree_dir = Path("/tmp/iree")
    clone_iree(get_iree_version(), iree_dir)

    for pyver in args.python_versions:
        python = Path(f"/opt/python/{pyver}/bin/python")
        if not python.exists():
            print(f"WARNING: {python} not found, skipping {pyver}")
            continue

        print(f"\n{'='*60}")
        print(f"Building wheel for {pyver}")
        print(f"{'='*60}")

        # Ensure build deps are installed for this Python.
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "scikit-build-core>=0.10",
                "nanobind>=2.9.0",
                "numpy>=1.26",
            ]
        )

        # Ensure patchelf is sane (0.17 series is broken).
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "patchelf==0.16.1.0",
            ]
        )

        # Find any pre-existing wheels matching this version to clean them.
        # Wheels may be version-specific (cp310-cp310) or abi3 (cp312-abi3).
        arch = platform.machine()
        cpver = pyver.split("-")[0]  # e.g. "cp312" from "cp312-cp312"
        for old in output_dir.glob(f"iree_tokenizer-*-{cpver}-*-linux_{arch}.whl"):
            old.unlink()
            print(f"  Cleaned old wheel: {old.name}")

        build_wheel(python, output_dir, iree_dir)

        if not args.no_repair:
            # Find the just-built generic wheel and repair it.
            # May be version-specific or abi3.
            generic_wheels = list(
                output_dir.glob(f"iree_tokenizer-*-{cpver}-*-linux_{arch}.whl")
            )
            if not generic_wheels:
                print(f"WARNING: No wheel found for {pyver} after build")
                continue
            for whl in generic_wheels:
                audit_wheel(whl, output_dir)

    print(f"\n{'='*60}")
    print("Build complete. Wheels:")
    for whl in sorted(output_dir.glob("*.whl")):
        print(f"  {whl.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build manylinux wheels for iree-tokenizer"
    )
    parser.add_argument(
        "--docker",
        default="docker",
        help="Container runtime command (docker, podman). Default: docker",
    )
    parser.add_argument(
        "--python-versions",
        nargs="+",
        default=DEFAULT_PYTHON_VERSIONS,
        help="CPython version tags (e.g. cp310-cp310 cp312-cp312)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "wheelhouse"),
        help="Output directory for wheels. Default: ./wheelhouse",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Override manylinux container image",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Skip auditwheel repair (for debugging)",
    )
    parser.add_argument(
        "--in-docker",
        action="store_true",
        help="Run in-container build (used internally by the trampoline)",
    )
    args = parser.parse_args()

    if args.in_docker:
        run_in_docker(args)
    else:
        run_on_host(args)


if __name__ == "__main__":
    main()
