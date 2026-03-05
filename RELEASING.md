# Releasing iree-tokenizer

## Prerequisites

- Push access to `iree-org/iree-tokenizer-py`

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

The `release.yml` workflow automatically triggers on the `vX.Y.Z` tag and:
1. Builds Linux and Windows wheels (via `build_packages.yml`)
2. Tests them across Python 3.10–3.14
3. Publishes to PyPI via OIDC trusted publishing

Monitor progress at:
https://github.com/iree-org/iree-tokenizer-py/actions/workflows/release.yml

### 3. Create a GitHub release (optional)

```bash
gh release create vX.Y.Z --repo iree-org/iree-tokenizer-py \
  --title "vX.Y.Z" --generate-notes
```

## Manual / emergency publishing

If the automated publish fails, you can download wheels from a CI run and
upload manually with twine:

```bash
pip install twine
python build_tools/publish_artifacts.py --run-id <RUN_ID>

# Or for TestPyPI:
python build_tools/publish_artifacts.py --run-id <RUN_ID> --test-pypi
```

Find the run ID via:
```bash
gh run list --workflow "Release" --repo iree-org/iree-tokenizer-py
```

## PyPI trusted publishing setup

The automated publish uses [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC). This is configured on pypi.org under the `iree-tokenizer` package
settings with:

- **Repository:** `iree-org/iree-tokenizer-py`
- **Workflow:** `release.yml`
- **Environment:** `pypi`

A corresponding `pypi` environment must exist in the GitHub repository
settings (Settings > Environments).

## Version scheme

- Release versions: `X.Y.Z` (PEP 440)
- Development versions: `X.Y.Z.dev0` (between releases on `main`)
- The version is stored in `version.json` and extracted by scikit-build-core
  at build time
