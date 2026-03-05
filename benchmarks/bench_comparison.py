#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Head-to-head benchmark: iree.tokenizer vs HF tokenizers vs tiktoken.

Downloads tokenizer models on first run (cached by huggingface_hub / tiktoken).
Prints a rich table to stdout and saves JSON results to benchmark_results.json.

Usage:
    python benchmarks/bench_comparison.py [--warmup N] [--iterations N]
"""

import argparse
import json
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

SHORT_TEXT = "The quick brown fox jumps over the lazy dog."
MEDIUM_TEXT = SHORT_TEXT * 20  # ~900 chars
LONG_TEXT = SHORT_TEXT * 500  # ~22K chars

CORPUS = {
    "short": SHORT_TEXT,
    "medium": MEDIUM_TEXT,
    "long": LONG_TEXT,
}


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------


def load_iree_tokenizer(model_id: str):
    """Load an IREE tokenizer from a HF model id."""
    from huggingface_hub import hf_hub_download

    from iree.tokenizer import Tokenizer

    path = hf_hub_download(model_id, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    return tok, tok.encode, tok.decode


def load_hf_tokenizer(model_id: str):
    """Load a HuggingFace tokenizers Tokenizer."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(model_id)

    def encode(text):
        return tok.encode(text).ids

    def decode(ids):
        return tok.decode(ids)

    return tok, encode, decode


def load_tiktoken(encoding_name: str):
    """Load a tiktoken encoding."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return enc, enc.encode, enc.decode


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def bench(fn, arg, warmup: int, iterations: int) -> dict:
    """Time fn(arg) and return stats."""
    for _ in range(warmup):
        fn(arg)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn(arg)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return {
        "p50_us": statistics.median(times) * 1e6,
        "p99_us": times[int(len(times) * 0.99)] * 1e6,
        "mean_us": statistics.mean(times) * 1e6,
        "iterations": iterations,
    }


def bench_encode(backends: dict, warmup: int, iterations: int) -> list[dict]:
    """Benchmark encode across backends and corpus sizes."""
    rows = []
    for corpus_name, text in CORPUS.items():
        for backend_name, (_, encode, _) in backends.items():
            stats = bench(encode, text, warmup, iterations)
            n_tokens = len(encode(text))
            stats["backend"] = backend_name
            stats["corpus"] = corpus_name
            stats["n_chars"] = len(text)
            stats["n_tokens"] = n_tokens
            stats["tokens_per_sec"] = n_tokens / (stats["mean_us"] / 1e6)
            rows.append(stats)
    return rows


def bench_decode(backends: dict, warmup: int, iterations: int) -> list[dict]:
    """Benchmark decode across backends."""
    rows = []
    for corpus_name, text in CORPUS.items():
        for backend_name, (_, encode, decode) in backends.items():
            ids = encode(text)
            stats = bench(decode, ids, warmup, iterations)
            stats["backend"] = backend_name
            stats["corpus"] = corpus_name
            stats["n_tokens"] = len(ids)
            rows.append(stats)
    return rows


def bench_batch(backends: dict, warmup: int, iterations: int) -> list[dict]:
    """Benchmark batch encode for iree.tokenizer only (others lack C batch)."""
    rows = []
    for batch_size in [1, 10, 100]:
        texts = [MEDIUM_TEXT] * batch_size
        for backend_name, (tok, encode, _) in backends.items():
            if backend_name == "iree":
                fn = lambda t: tok.encode_batch(t)
            elif backend_name == "hf":
                fn = lambda t: [tok.encode(s).ids for s in t]
            elif backend_name == "tiktoken":
                fn = lambda t: [tok.encode(s) for s in t]
            else:
                continue
            stats = bench(fn, texts, warmup, iterations)
            total_tokens = sum(len(encode(t)) for t in texts)
            stats["backend"] = backend_name
            stats["batch_size"] = batch_size
            stats["total_tokens"] = total_tokens
            stats["tokens_per_sec"] = total_tokens / (stats["mean_us"] / 1e6)
            rows.append(stats)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(title: str, rows: list[dict], columns: list[str]):
    """Print a rich table."""
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title)
    for col in columns:
        table.add_column(col, justify="right" if col != "backend" else "left")
    for row in rows:
        table.add_row(*[fmt_val(row.get(c, "")) for c in columns])
    Console().print(table)


def fmt_val(v):
    if isinstance(v, float):
        if v > 1e6:
            return f"{v:.0f}"
        return f"{v:.1f}"
    return str(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate_backends(backends: dict) -> bool:
    """Cross-backend correctness checks. Returns True if all pass."""
    ok = True
    backend_names = sorted(backends.keys())

    # Check 1: Cross-backend agreement on encode output.
    print("\n[validate] Cross-backend agreement...")
    for corpus_name, text in CORPUS.items():
        results = {}
        for name in backend_names:
            _, encode, _ = backends[name]
            results[name] = encode(text)
        ref_name = backend_names[0]
        ref_ids = results[ref_name]
        for name in backend_names[1:]:
            if results[name] != ref_ids:
                print(
                    f"  FAIL: {name} != {ref_name} on corpus '{corpus_name}' "
                    f"({len(results[name])} vs {len(ref_ids)} tokens)"
                )
                ok = False
            else:
                print(f"  OK: {ref_name} == {name} on '{corpus_name}'")

    # Check 2: Encode-decode roundtrip.
    print("[validate] Encode-decode roundtrip...")
    for corpus_name, text in CORPUS.items():
        for name in backend_names:
            _, encode, decode = backends[name]
            ids = encode(text)
            decoded = decode(ids)
            if decoded != text:
                print(
                    f"  FAIL: {name} roundtrip on '{corpus_name}': "
                    f"decoded {len(decoded)} chars vs original {len(text)}"
                )
                ok = False
            else:
                print(f"  OK: {name} roundtrip on '{corpus_name}'")

    # Check 3: Batch consistency (iree only).
    if "iree" in backends:
        print("[validate] Batch consistency (iree)...")
        tok, encode, _ = backends["iree"]
        for corpus_name, text in CORPUS.items():
            single_ids = encode(text)
            batch_ids = tok.encode_batch([text])[0]
            if batch_ids != single_ids:
                print(
                    f"  FAIL: encode_batch != encode on '{corpus_name}' "
                    f"({len(batch_ids)} vs {len(single_ids)} tokens)"
                )
                ok = False
            else:
                print(f"  OK: batch consistency on '{corpus_name}'")

    return ok


def main():
    parser = argparse.ArgumentParser(description="iree-tokenizer benchmarks")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--model",
        default="openai-community/gpt2",
        help="HF model id for iree/hf backends",
    )
    parser.add_argument(
        "--tiktoken-encoding",
        default="gpt2",
        help="tiktoken encoding name",
    )
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run cross-backend correctness checks after benchmarking",
    )
    args = parser.parse_args()

    if args.validate:
        print(
            "WARNING: --validate enabled; correctness checks will run after benchmarks.\n"
        )

    backends = {}

    # Load backends, skip gracefully if unavailable.
    try:
        backends["iree"] = load_iree_tokenizer(args.model)
        print(f"Loaded iree.tokenizer from {args.model}")
    except Exception as e:
        print(f"Skipping iree: {e}", file=sys.stderr)

    try:
        backends["hf"] = load_hf_tokenizer(args.model)
        print(f"Loaded HF tokenizers from {args.model}")
    except Exception as e:
        print(f"Skipping hf: {e}", file=sys.stderr)

    try:
        backends["tiktoken"] = load_tiktoken(args.tiktoken_encoding)
        print(f"Loaded tiktoken encoding {args.tiktoken_encoding}")
    except Exception as e:
        print(f"Skipping tiktoken: {e}", file=sys.stderr)

    if not backends:
        print("No backends available!", file=sys.stderr)
        sys.exit(1)

    print()

    # Run benchmarks.
    encode_rows = bench_encode(backends, args.warmup, args.iterations)
    print_table(
        "Encode Latency",
        encode_rows,
        [
            "backend",
            "corpus",
            "n_chars",
            "n_tokens",
            "p50_us",
            "p99_us",
            "tokens_per_sec",
        ],
    )

    decode_rows = bench_decode(backends, args.warmup, args.iterations)
    print_table(
        "Decode Latency",
        decode_rows,
        ["backend", "corpus", "n_tokens", "p50_us", "p99_us"],
    )

    batch_rows = bench_batch(backends, args.warmup, args.iterations)
    print_table(
        "Batch Encode",
        batch_rows,
        ["backend", "batch_size", "total_tokens", "p50_us", "p99_us", "tokens_per_sec"],
    )

    # Save results.
    results = {
        "encode": encode_rows,
        "decode": decode_rows,
        "batch": batch_rows,
        "config": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "model": args.model,
            "validated": args.validate,
        },
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Run validation after saving results (no timing pollution).
    if args.validate:
        if not validate_backends(backends):
            print("\nVALIDATION FAILED", file=sys.stderr)
            sys.exit(2)
        print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
