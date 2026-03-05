# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib

import numpy as np
import pytest

from iree.tokenizer import Tokenizer

TESTDATA = pathlib.Path(__file__).parent / "testdata"
TIKTOKEN_PATH = TESTDATA / "tiktoken_gpt2.tiktoken"


def test_from_tiktoken():
    tok = Tokenizer.from_tiktoken(str(TIKTOKEN_PATH), encoding="gpt2")
    assert tok.vocab_size == 261


def test_from_tiktoken_str():
    data = TIKTOKEN_PATH.read_text()
    tok = Tokenizer.from_tiktoken_str(data, encoding="gpt2")
    assert tok.vocab_size == 261


def test_from_tiktoken_buffer():
    data = TIKTOKEN_PATH.read_bytes()
    tok = Tokenizer.from_tiktoken_buffer(data, encoding="gpt2")
    assert tok.vocab_size == 261


def test_tiktoken_invalid_encoding():
    with pytest.raises(ValueError, match="Unknown tiktoken encoding"):
        Tokenizer.from_tiktoken(str(TIKTOKEN_PATH), encoding="nonexistent")


def test_tiktoken_model_type(tiktoken_tokenizer):
    assert tiktoken_tokenizer.model_type == "BPE"


def test_tiktoken_encode_decode_roundtrip(tiktoken_tokenizer):
    text = "Hello world"
    ids = tiktoken_tokenizer.encode(text)
    assert ids == [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
    decoded = tiktoken_tokenizer.decode(ids)
    assert decoded == text


def test_tiktoken_encode_empty(tiktoken_tokenizer):
    ids = tiktoken_tokenizer.encode("")
    assert ids == []


def test_tiktoken_encode_batch(tiktoken_tokenizer):
    texts = ["Hello", "world"]
    batch = tiktoken_tokenizer.encode_batch(texts)
    assert batch[0] == [72, 101, 108, 108, 111]
    assert batch[1] == [119, 111, 114, 108, 100]


def test_tiktoken_encode_to_array(tiktoken_tokenizer):
    arr = tiktoken_tokenizer.encode_to_array("Hello world")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    ids = tiktoken_tokenizer.encode("Hello world")
    np.testing.assert_array_equal(arr, ids)
