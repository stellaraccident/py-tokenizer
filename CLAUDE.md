# CLAUDE.md — iree-tokenizer-py

Python bindings for the IREE high-performance C tokenizer, published as
`iree-tokenizer` on PyPI.

## Repository layout

```
src/bindings/          C++ nanobind bindings (tokenizer.cc, streaming.cc, encoding.cc)
src/iree/tokenizer/   Python package (__init__.py, cli.py)
tests/                 pytest tests (run with PYTHONPATH=src pytest tests/ -v)
benchmarks/            Comparison benchmarks vs tiktoken / HF tokenizers
build_tools/           Release scripts, Docker-based manylinux builder
.github/workflows/     CI (ci.yml) and wheel builds (build_packages.yml)
```

## Key files

- `version.json` — single source of truth for package version and IREE dep version
- `pyproject.toml` — build config (scikit-build-core + nanobind)
- `CMakeLists.txt` — builds the `_iree_tokenizer` native extension
- `PYPI_README.md` — text-only README used on PyPI (no images)
- `README.md` — full README with benchmark charts for GitHub

## Build and test

```bash
# Development build (requires IREE source checkout)
cmake -B build -G Ninja -DIREE_SOURCE_DIR=/path/to/iree
cmake --build build
ln -s build/_iree_tokenizer*.so src/iree/tokenizer/
PYTHONPATH=src pytest tests/ -v

# ASAN build (requires Clang)
cmake -B build-asan -G Ninja \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_TOKENIZER_ENABLE_ASAN=ON
cmake --build build-asan
ln -sf build-asan/_iree_tokenizer*.so src/iree/tokenizer/
LD_PRELOAD=$(clang++ -print-file-name=libclang_rt.asan-x86_64.so) \
  ASAN_OPTIONS=detect_leaks=0 PYTHONPATH=src pytest tests/ -v

# pip install from source
IREE_SOURCE_DIR=/path/to/iree pip install -e ".[test]"
```

## Bindings architecture

- **nanobind** with `NB_STATIC LTO STABLE_ABI FREE_THREADED` flags
- Stable ABI (`cp312-abi3`) — one wheel covers Python 3.12+
- `FREE_THREADED` — PEP 703 compatible; mutable stream objects use `nb::lock_self()`
- Memory ownership: numpy arrays use `unique_ptr` + `nb::capsule` for leak-safe handoff
- Streams hold raw `const iree_tokenizer_t*` kept alive via `nb::keep_alive<0,1>()`

## Conventions

- All commits to `main` require `Signed-off-by` (DCO)
- Pre-commit hooks: black, clang-format, trailing whitespace
- C++ warnings: `-Wall -Wextra -Wshadow` / `/W4`
- Test tokenizer fixture: `tests/testdata/bpe_bytelevel_minimal.json`
- `conftest.py` provides `bpe_tokenizer` fixture used across test files

IMPORTANT: Agents must do commit incremental work in topic branches and raise PRs unless if instructed case-by-case to do differently by the user.

## Release process

See `RELEASING.md`. Key script: `build_tools/make_release.py`.

IMPORTANT: Agents must not initiate releasing activities without express permission and monitoring from the user.

## GitHub

Repository: https://github.com/iree-org/iree-tokenizer-py
