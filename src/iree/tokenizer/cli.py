#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Streaming tokenizer CLI.

Pipe-compatible encode/decode with JSONL output and live throughput on stderr.

Usage:
    echo "Hello world" | iree-tokenizer-python encode -t tokenizer.json
    cat corpus.txt | iree-tokenizer-python encode -t tok.json | iree-tokenizer-python decode -t tok.json
    iree-tokenizer-python info -t tokenizer.json

Note that this tool illustrates streaming processing but the overhead of JSON
processing is expensive and skews throughput. Treat this as an example of how
to operate the streaming API vs a benchmarking tool or a tool expected to
achieve maximum throughput.
"""

import argparse
import json
import sys
import time

from iree.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------


def read_lines(stream):
    for line in stream:
        stripped = line.rstrip("\n")
        if stripped:
            yield stripped


def read_paragraphs(stream):
    buf = []
    for line in stream:
        stripped = line.rstrip("\n")
        if stripped == "" and buf:
            yield "\n".join(buf)
            buf = []
        else:
            buf.append(stripped)
    if buf:
        yield "\n".join(buf)


def read_whole(stream):
    text = stream.read()
    if text.strip():
        yield text


# ---------------------------------------------------------------------------
# Input line parsing (for chaining support)
# ---------------------------------------------------------------------------


def parse_encode_input(line):
    """Parse a line for the encode command.

    Returns the text to encode. Accepts:
    - JSON object with "text" field (from decode output)
    - Plain text
    """
    try:
        obj = json.loads(line)
        if isinstance(obj, dict) and "text" in obj:
            return obj["text"]
    except (json.JSONDecodeError, ValueError):
        pass
    return line


def parse_decode_input(line):
    """Parse a line for the decode command.

    Returns a list of token IDs. Accepts:
    - JSON object with "ids" field (from encode output)
    - JSON array of ints
    """
    obj = json.loads(line)
    if isinstance(obj, dict) and "ids" in obj:
        return obj["ids"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Expected JSON object with 'ids' or JSON array, got: {line!r}")


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------


def _fmt_rate(tps):
    if tps >= 1_000_000:
        return f"{tps / 1_000_000:.2f}M"
    if tps >= 1_000:
        return f"{tps / 1_000:.1f}K"
    return f"{tps:.0f}"


class ProgressTracker:
    def __init__(self, mode, enabled):
        self.mode = mode
        self.enabled = enabled
        self.is_tty = sys.stderr.isatty()
        self.start = time.perf_counter()
        self.chunks = 0
        self.total_tokens = 0
        self.total_bytes = 0
        self._last_display = 0.0

    def update(self, n_tokens, n_bytes):
        self.chunks += 1
        self.total_tokens += n_tokens
        self.total_bytes += n_bytes
        if self.enabled and self.is_tty:
            now = time.perf_counter()
            if now - self._last_display >= 1.0:
                self._display_live()
                self._last_display = now

    def _display_live(self):
        elapsed = time.perf_counter() - self.start
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        mb = self.total_bytes / 1_000_000
        line = (
            f"\r[{self.mode}] {self.chunks:,} chunks | "
            f"{self.total_tokens:,} tokens | "
            f"{_fmt_rate(tps)} tok/s | {mb:.1f} MB"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def finalize(self):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        mb = self.total_bytes / 1_000_000
        if self.is_tty:
            sys.stderr.write("\r\033[K")
        sys.stderr.write(
            f"[{self.mode}] done: {self.chunks:,} chunks | "
            f"{self.total_tokens:,} tokens in {elapsed:.3f}s "
            f"({_fmt_rate(tps)} tok/s) | {mb:.1f} MB\n"
        )
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_encode(tok, args):
    reader = {
        "line": read_lines,
        "paragraph": read_paragraphs,
        "whole": read_whole,
    }[args.input_mode]
    progress = ProgressTracker("encode", not args.no_progress)
    compact = args.compact
    rich = args.rich
    add_special = args.add_special_tokens
    out = sys.stdout

    try:
        for seq, raw in enumerate(reader(sys.stdin)):
            text = parse_encode_input(raw)
            t0 = time.perf_counter()
            if rich:
                enc = tok.encode_rich(
                    text, add_special_tokens=add_special, track_offsets=True
                )
                ids = enc.ids.tolist()
                offsets = enc.offsets.tolist()
            else:
                ids = tok.encode(text, add_special_tokens=add_special)
                offsets = None
            elapsed_ms = (time.perf_counter() - t0) * 1000

            record = {
                "seq": seq,
                "text": text,
                "ids": ids,
                "n_tokens": len(ids),
            }
            if not compact:
                record["input_chars"] = len(text)
                record["elapsed_ms"] = round(elapsed_ms, 3)
            if offsets is not None:
                record["offsets"] = offsets

            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            out.flush()
            progress.update(len(ids), len(text))
    except KeyboardInterrupt:
        pass
    finally:
        progress.finalize()


def cmd_decode(tok, args):
    progress = ProgressTracker("decode", not args.no_progress)
    compact = args.compact
    skip_special = args.skip_special_tokens
    out = sys.stdout

    try:
        for seq, line in enumerate(read_lines(sys.stdin)):
            ids = parse_decode_input(line)
            t0 = time.perf_counter()
            text = tok.decode(ids, skip_special_tokens=skip_special)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            record = {
                "seq": seq,
                "ids": ids,
                "text": text,
                "n_tokens": len(ids),
            }
            if not compact:
                record["elapsed_ms"] = round(elapsed_ms, 3)

            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            out.flush()
            progress.update(len(ids), len(text))
    except KeyboardInterrupt:
        pass
    finally:
        progress.finalize()


def cmd_info(tok):
    info = {
        "vocab_size": tok.vocab_size,
        "model_type": tok.model_type,
    }
    for attr in (
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
        "pad_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        val = getattr(tok, attr, None)
        if val is not None:
            info[attr] = val
    json.dump(info, sys.stdout, indent=2)
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _add_common_args(subparser):
    subparser.add_argument(
        "-t", "--tokenizer", help="Path to tokenizer.json or .tiktoken file"
    )
    subparser.add_argument(
        "--tokenizer-json", help="Tokenizer JSON string (alternative to -t)"
    )
    subparser.add_argument(
        "--encoding",
        help="Tiktoken encoding name (required for .tiktoken files). "
        "Supported: cl100k_base, o200k_base, o200k_harmony, r50k_base, "
        "gpt2, p50k_base, p50k_edit.",
    )


def make_parser():
    parser = argparse.ArgumentParser(
        prog="iree-tokenizer-python",
        description="Streaming tokenizer CLI backed by IREE",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode text to token IDs")
    _add_common_args(enc)
    enc.add_argument(
        "--input-mode",
        choices=["line", "paragraph", "whole"],
        default="line",
    )
    enc.add_argument("--rich", action="store_true", help="Include byte offsets")
    enc.add_argument("--add-special-tokens", action="store_true")
    enc.add_argument("--compact", action="store_true", help="Omit timing/size fields")
    enc.add_argument(
        "--no-progress", action="store_true", help="Suppress stderr progress"
    )

    dec = sub.add_parser("decode", help="Decode token IDs to text")
    _add_common_args(dec)
    dec.add_argument("--skip-special-tokens", action="store_true")
    dec.add_argument("--compact", action="store_true", help="Omit timing fields")
    dec.add_argument(
        "--no-progress", action="store_true", help="Suppress stderr progress"
    )

    info = sub.add_parser("info", help="Print tokenizer metadata")
    _add_common_args(info)

    return parser


def _load_tokenizer(args):
    if args.tokenizer:
        if args.tokenizer.endswith(".tiktoken"):
            if not args.encoding:
                print(
                    "Error: --encoding required for .tiktoken files",
                    file=sys.stderr,
                )
                sys.exit(1)
            return Tokenizer.from_tiktoken(args.tokenizer, encoding=args.encoding)
        return Tokenizer.from_file(args.tokenizer)
    if args.tokenizer_json:
        return Tokenizer.from_str(args.tokenizer_json)
    print("Error: --tokenizer or --tokenizer-json required", file=sys.stderr)
    sys.exit(1)


def main():
    parser = make_parser()
    args = parser.parse_args()
    tok = _load_tokenizer(args)

    if args.command == "encode":
        cmd_encode(tok, args)
    elif args.command == "decode":
        cmd_decode(tok, args)
    elif args.command == "info":
        cmd_info(tok)


if __name__ == "__main__":
    main()
