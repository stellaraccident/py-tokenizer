# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import pytest

from iree.tokenizer import Tokenizer

TESTDATA = pathlib.Path(__file__).parent / "testdata"
BPE_PATH = TESTDATA / "bpe_bytelevel_minimal.json"


def test_from_file():
    tok = Tokenizer.from_file(str(BPE_PATH))
    assert tok.vocab_size == 112


def test_from_str():
    json_str = BPE_PATH.read_text()
    tok = Tokenizer.from_str(json_str)
    assert tok.vocab_size == 112


def test_from_buffer():
    data = BPE_PATH.read_bytes()
    tok = Tokenizer.from_buffer(data)
    assert tok.vocab_size == 112


def test_from_file_not_found():
    with pytest.raises(ValueError, match="Cannot open"):
        Tokenizer.from_file("/nonexistent/path.json")


def test_from_str_invalid_json():
    with pytest.raises(Exception):
        Tokenizer.from_str("not valid json")


def test_repr(bpe_tokenizer):
    r = repr(bpe_tokenizer)
    assert "BPE" in r
    assert "vocab_size" in r
