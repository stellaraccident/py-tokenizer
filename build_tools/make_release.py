#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Create a release: update version, commit, tag, optionally bump to dev.

Usage:
    # Release 0.1.0, then bump to 0.2.0.dev0:
    python build_tools/make_release.py --version 0.1.0 --bump-dev

    # Release 0.1.0 without post-release bump:
    python build_tools/make_release.py --version 0.1.0

    # Dry run (show what would happen):
    python build_tools/make_release.py --version 0.1.0 --bump-dev --dry-run

After running, manually:
    git push origin main --tags
    python build_tools/publish_artifacts.py --latest
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "version.json"


def run(cmd: list[str], dry_run: bool = False):
    print(f"+ {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def read_version_json() -> dict:
    with open(VERSION_FILE) as f:
        return json.load(f)


def write_version_json(data: dict, dry_run: bool = False):
    content = json.dumps(data, indent=2) + "\n"
    print(f"  version.json: package-version = {data['package-version']}")
    if not dry_run:
        with open(VERSION_FILE, "w") as f:
            f.write(content)


def validate_version(version: str):
    """Validate PEP 440 release version (X.Y.Z)."""
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        sys.exit(f"Invalid version '{version}': expected X.Y.Z (e.g., 0.1.0)")


def next_dev_version(version: str) -> str:
    """Bump minor version and append .dev0: 0.1.0 -> 0.2.0.dev0."""
    parts = version.split(".")
    parts[1] = str(int(parts[1]) + 1)
    parts[2] = "0"
    return ".".join(parts) + ".dev0"


def check_clean_tree(dry_run: bool):
    if dry_run:
        return
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.stdout.strip():
        sys.exit("Working tree is not clean. Commit or stash changes first.")


def main():
    parser = argparse.ArgumentParser(description="Create an iree-tokenizer release")
    parser.add_argument(
        "--version", required=True, help="Release version (e.g., 0.1.0)"
    )
    parser.add_argument(
        "--bump-dev",
        action="store_true",
        help="After tagging, bump to next dev version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    args = parser.parse_args()

    validate_version(args.version)
    check_clean_tree(args.dry_run)

    data = read_version_json()
    tag = f"v{args.version}"

    # Step 1: Set release version.
    print(f"\n=== Setting version to {args.version} ===")
    data["package-version"] = args.version
    write_version_json(data, args.dry_run)
    run(["git", "add", "version.json"], args.dry_run)
    run(["git", "commit", "-m", f"Release {tag}"], args.dry_run)

    # Step 2: Create tag.
    print(f"\n=== Creating tag {tag} ===")
    run(["git", "tag", tag], args.dry_run)

    # Step 3: Optionally bump to next dev version.
    if args.bump_dev:
        dev = next_dev_version(args.version)
        print(f"\n=== Bumping to {dev} ===")
        data["package-version"] = dev
        write_version_json(data, args.dry_run)
        run(["git", "add", "version.json"], args.dry_run)
        run(["git", "commit", "-m", f"Bump to {dev}"], args.dry_run)

    # Done.
    print(f"\n{'='*60}")
    print("Done! Next steps:")
    print(f"  git push origin main --tags")
    print(f"  # Wait for CI to build wheels, then:")
    print(f"  python build_tools/publish_artifacts.py --latest")
    if args.dry_run:
        print("\n(dry run — no changes were made)")


if __name__ == "__main__":
    main()
