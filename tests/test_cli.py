# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration tests for the iree-tokenizer-python CLI."""

import json
import pathlib
import subprocess
import sys

import pytest

TOKENIZER = str(
    pathlib.Path(__file__).parent / "testdata" / "bpe_bytelevel_minimal.json"
)
CLI = [sys.executable, "-m", "iree.tokenizer.cli"]


def run_cli(*args, input_text=None):
    result = subprocess.run(
        [*CLI, *args],
        input=input_text,
        capture_output=True,
        text=True,
    )
    return result


def parse_jsonl(stdout):
    """Parse JSONL output, skipping empty lines."""
    records = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


class TestEncode:
    def test_encode_single_line(self):
        r = run_cli("encode", "-t", TOKENIZER, "--no-progress", input_text="Hello\n")
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert len(records) == 1
        assert "ids" in records[0]
        assert "text" in records[0]
        assert records[0]["text"] == "Hello"
        assert records[0]["seq"] == 0
        assert records[0]["n_tokens"] == len(records[0]["ids"])

    def test_encode_multiple_lines(self):
        r = run_cli(
            "encode", "-t", TOKENIZER, "--no-progress", input_text="Hello\nworld\n"
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert len(records) == 2
        assert records[0]["seq"] == 0
        assert records[1]["seq"] == 1

    def test_encode_compact(self):
        r = run_cli(
            "encode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--compact",
            input_text="Hello\n",
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert "elapsed_ms" not in records[0]
        assert "input_chars" not in records[0]
        assert "ids" in records[0]
        assert "text" in records[0]

    def test_encode_rich(self):
        r = run_cli(
            "encode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--rich",
            input_text="Hello\n",
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert "offsets" in records[0]

    def test_encode_paragraph_mode(self):
        r = run_cli(
            "encode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--input-mode",
            "paragraph",
            input_text="line one\nline two\n\nline three\n",
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert len(records) == 2
        assert "line one\nline two" in records[0]["text"]
        assert "line three" in records[1]["text"]

    def test_encode_whole_mode(self):
        r = run_cli(
            "encode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--input-mode",
            "whole",
            input_text="Hello\nworld\n",
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert len(records) == 1

    def test_encode_empty_input(self):
        r = run_cli("encode", "-t", TOKENIZER, "--no-progress", input_text="")
        assert r.returncode == 0
        assert r.stdout.strip() == ""


class TestDecode:
    def test_decode_json_array(self):
        r = run_cli(
            "decode", "-t", TOKENIZER, "--no-progress", input_text="[72, 101]\n"
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert len(records) == 1
        assert "text" in records[0]
        assert "ids" in records[0]
        assert records[0]["ids"] == [72, 101]

    def test_decode_json_object(self):
        r = run_cli(
            "decode",
            "-t",
            TOKENIZER,
            "--no-progress",
            input_text='{"seq":0,"ids":[72,101],"text":"Hello","n_tokens":2}\n',
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert len(records) == 1
        assert "text" in records[0]

    def test_decode_compact(self):
        r = run_cli(
            "decode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--compact",
            input_text="[72, 101]\n",
        )
        assert r.returncode == 0
        records = parse_jsonl(r.stdout)
        assert "elapsed_ms" not in records[0]


class TestChaining:
    def test_encode_then_decode_roundtrip(self):
        """Encode output feeds directly into decode."""
        enc = run_cli(
            "encode", "-t", TOKENIZER, "--no-progress", input_text="Hello world\n"
        )
        assert enc.returncode == 0
        dec = run_cli("decode", "-t", TOKENIZER, "--no-progress", input_text=enc.stdout)
        assert dec.returncode == 0
        dec_records = parse_jsonl(dec.stdout)
        assert len(dec_records) == 1
        assert dec_records[0]["text"] == "Hello world"

    def test_decode_then_encode_roundtrip(self):
        """Decode output feeds directly into encode."""
        # First encode to get valid IDs.
        enc1 = run_cli(
            "encode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--compact",
            input_text="Hello world\n",
        )
        assert enc1.returncode == 0
        original_ids = parse_jsonl(enc1.stdout)[0]["ids"]

        # Decode those IDs.
        dec = run_cli(
            "decode", "-t", TOKENIZER, "--no-progress", input_text=enc1.stdout
        )
        assert dec.returncode == 0

        # Re-encode from decode output.
        enc2 = run_cli(
            "encode",
            "-t",
            TOKENIZER,
            "--no-progress",
            "--compact",
            input_text=dec.stdout,
        )
        assert enc2.returncode == 0
        roundtrip_ids = parse_jsonl(enc2.stdout)[0]["ids"]
        assert roundtrip_ids == original_ids

    def test_multi_line_roundtrip(self):
        """Multiple lines survive encode → decode chain."""
        text = "Hello world\nThe quick brown fox\n"
        enc = run_cli("encode", "-t", TOKENIZER, "--no-progress", input_text=text)
        assert enc.returncode == 0
        dec = run_cli("decode", "-t", TOKENIZER, "--no-progress", input_text=enc.stdout)
        assert dec.returncode == 0
        dec_records = parse_jsonl(dec.stdout)
        assert len(dec_records) == 2
        assert dec_records[0]["text"] == "Hello world"
        assert dec_records[1]["text"] == "The quick brown fox"


class TestInfo:
    def test_info(self):
        r = run_cli("info", "-t", TOKENIZER)
        assert r.returncode == 0
        info = json.loads(r.stdout)
        assert "vocab_size" in info
        assert "model_type" in info
        assert isinstance(info["vocab_size"], int)


class TestProgress:
    def test_no_progress_suppresses_stderr(self):
        r = run_cli("encode", "-t", TOKENIZER, "--no-progress", input_text="Hello\n")
        assert r.returncode == 0
        assert r.stderr == ""

    def test_default_progress_writes_stderr(self):
        r = run_cli("encode", "-t", TOKENIZER, input_text="Hello\n")
        assert r.returncode == 0
        # When not a TTY, only the final summary line is printed.
        assert "[encode] done:" in r.stderr
