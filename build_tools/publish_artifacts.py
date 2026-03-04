#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Download wheel artifacts from a GitHub Actions run and publish to PyPI.

Requires: gh (GitHub CLI), twine

Usage:
    # Download + publish to TestPyPI:
    python build_tools/publish_artifacts.py --run-id 12345 --test-pypi

    # Download + publish to real PyPI:
    python build_tools/publish_artifacts.py --run-id 12345

    # Just download, don't upload:
    python build_tools/publish_artifacts.py --run-id 12345 --dry-run

    # Use latest run from main:
    python build_tools/publish_artifacts.py --latest
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "stellaraccident/py-tokenizer"


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def get_latest_run_id() -> str:
    result = run(
        [
            "gh",
            "run",
            "list",
            "--repo",
            REPO,
            "--workflow",
            "build_packages.yml",
            "--branch",
            "main",
            "--status",
            "success",
            "--limit",
            "1",
            "--json",
            "databaseId",
            "-q",
            ".[0].databaseId",
        ],
        capture_output=True,
        text=True,
    )
    run_id = result.stdout.strip()
    if not run_id:
        sys.exit("No successful build_packages runs found on main")
    return run_id


def download_artifacts(run_id: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    run(
        [
            "gh",
            "run",
            "download",
            run_id,
            "--repo",
            REPO,
            "--name",
            "wheels-linux-x86_64",
            "--dir",
            str(dest),
        ]
    )


def upload_wheels(wheel_dir: Path, test_pypi: bool, dry_run: bool):
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        sys.exit(f"No .whl files found in {wheel_dir}")

    print(f"\nWheels to upload ({len(wheels)}):")
    for w in wheels:
        print(f"  {w.name}")

    if dry_run:
        print("\n--dry-run: skipping upload")
        return

    cmd = ["twine", "upload"]
    if test_pypi:
        cmd += ["--repository", "testpypi"]
    cmd += [str(w) for w in wheels]
    run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Download and publish iree-tokenizer wheels from CI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="GitHub Actions run ID")
    group.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest successful run on main",
    )
    parser.add_argument(
        "--test-pypi",
        action="store_true",
        help="Upload to TestPyPI instead of PyPI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download artifacts but don't upload",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to download wheels into (default: temp dir)",
    )
    args = parser.parse_args()

    run_id = args.run_id or get_latest_run_id()
    print(f"Using run ID: {run_id}")

    if args.output_dir:
        dest = Path(args.output_dir)
        download_artifacts(run_id, dest)
        upload_wheels(dest, args.test_pypi, args.dry_run)
    else:
        with tempfile.TemporaryDirectory(prefix="iree-tokenizer-wheels-") as tmpdir:
            dest = Path(tmpdir)
            download_artifacts(run_id, dest)
            upload_wheels(dest, args.test_pypi, args.dry_run)


if __name__ == "__main__":
    main()
