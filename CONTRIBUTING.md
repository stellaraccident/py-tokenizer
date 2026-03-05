# Contributing to iree-tokenizer

## Getting in touch

- **GitHub Issues** — feature requests, bugs, and work tracking
- **[IREE Discord server](https://discord.gg/wEWh6Z9nMV)** — daily
  development discussions with the core team and collaborators
- **[iree-technical-discussion email list](https://groups.google.com/g/iree-technical-discussion)** —
  general and low-priority discussion

## Community guidelines

This project follows the
[LF Projects code of conduct](https://lfprojects.org/policies/code-of-conduct/).

## How to contribute

### Start with an issue

This project is maintained primarily via AI-assisted development. Because of
this workflow, **non-trivial changes are best started as a detailed issue**
rather than a cold PR.

A good issue describes the *desired end state* — what should change, why, and
any constraints — rather than prescribing an implementation. Think of it as a
short spec. This lets the maintainers (and their agents) explore the design
space and produce a well-integrated solution.

For small, self-contained fixes (typos, obvious bugs, doc corrections), a
direct PR is fine.

### Pull request guidelines

- All commits must include a `Signed-off-by` line (DCO). Use `git commit
  --signoff` or add it manually.
- Ensure pre-commit hooks pass (`pre-commit run --all-files`). The repo uses
  black, clang-format, and standard file hygiene checks.
- Include tests for new functionality. The test suite runs under ASAN in CI.
- Keep PRs focused — one logical change per PR.

### Development setup

```bash
# Clone and build from source
git clone https://github.com/iree-org/iree.git /path/to/iree
git clone https://github.com/iree-org/iree-tokenizer-py.git
cd iree-tokenizer-py
IREE_SOURCE_DIR=/path/to/iree pip install -e ".[test]"

# Run tests
pytest tests/ -v
```

See `README.md` for additional build options (ASAN, native arch, CMake).

## License

By contributing, you agree that your contributions will be licensed under the
Apache License v2.0 with LLVM Exceptions. See [LICENSE](LICENSE) for details.
