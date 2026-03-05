# Releasing iree-tokenizer

## Prerequisites

- Push access to `iree-org/iree-tokenizer-py`
- `twine` installed (`pip install twine`)
- PyPI credentials configured (or a PyPI API token)

## Release process

### 1. Run the release script

From a clean `main` branch:

```bash
python build_tools/make_release.py --version X.Y.Z --bump-dev
```

This will:
- Set `package-version` to `X.Y.Z` in `version.json` and commit
- Create a `vX.Y.Z` git tag
- Bump to `X.(Y+1).0.dev0` and commit

Both commits include `Signed-off-by` (DCO required by iree-org).

Use `--dry-run` first to verify:
```bash
python build_tools/make_release.py --version X.Y.Z --bump-dev --dry-run
```

### 2. Push

```bash
git push origin main --tags
```

### 3. Build wheels

**Current workaround:** The Build Packages workflow does not yet trigger on
tags. You must manually dispatch it:

1. Go to **Actions > Build Packages** on GitHub
2. Click **Run workflow**
3. Select the `vX.Y.Z` tag as the branch/ref
4. Wait for the build to complete

### 4. Publish to PyPI

Download the wheel artifacts from the tag build run and upload:

```bash
# Download artifacts from the run and upload to PyPI
python build_tools/publish_artifacts.py --run-id <RUN_ID>

# Or for a test run against TestPyPI first:
python build_tools/publish_artifacts.py --run-id <RUN_ID> --test-pypi
```

You can find the run ID in the Actions UI URL or via:
```bash
gh run list --workflow "Build Packages" --repo iree-org/iree-tokenizer-py
```

### 5. Create a GitHub release (optional)

```bash
gh release create vX.Y.Z --repo iree-org/iree-tokenizer-py \
  --title "vX.Y.Z" --generate-notes
```

## Known issues

The release workflow has several manual steps that should be automated. See
https://github.com/iree-org/iree-tokenizer-py/issues/9 for planned
improvements including tag-triggered builds and automated PyPI publishing
via OIDC trusted publishing.

## Version scheme

- Release versions: `X.Y.Z` (PEP 440)
- Development versions: `X.Y.Z.dev0` (between releases on `main`)
- The version is stored in `version.json` and extracted by scikit-build-core
  at build time
