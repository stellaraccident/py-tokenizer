# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from iree.tokenizer import Tokenizer


def test_encode_basic(bpe_tokenizer):
    ids = bpe_tokenizer.encode("Hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert ids == [39, 68, 105, 110]


def test_encode_empty(bpe_tokenizer):
    ids = bpe_tokenizer.encode("")
    assert isinstance(ids, list)
    assert ids == []


def test_encode_roundtrip(bpe_tokenizer):
    text = "Hello world"
    ids = bpe_tokenizer.encode(text)
    decoded = bpe_tokenizer.decode(ids)
    assert decoded == text


def test_encode_to_array(bpe_tokenizer):
    arr = bpe_tokenizer.encode_to_array("Hello world")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int32
    assert arr.ndim == 1
    np.testing.assert_array_equal(arr, [39, 68, 105, 110])
    # Should match list encode.
    ids = bpe_tokenizer.encode("Hello world")
    np.testing.assert_array_equal(arr, ids)


def test_encode_rich_with_offsets(bpe_tokenizer):
    enc = bpe_tokenizer.encode_rich("Hello world", track_offsets=True)
    assert enc.ids.dtype == np.int32
    np.testing.assert_array_equal(enc.ids, [39, 68, 105, 110])
    assert enc.offsets is not None
    assert enc.offsets.shape == (len(enc), 2)
    assert enc.offsets.dtype == np.uint64
    np.testing.assert_array_equal(enc.offsets, [[0, 1], [1, 2], [2, 5], [5, 11]])
    assert enc.type_ids.dtype == np.uint8
    np.testing.assert_array_equal(enc.type_ids, [0, 0, 0, 0])


def test_encode_rich_without_offsets(bpe_tokenizer):
    enc = bpe_tokenizer.encode_rich("Hello world", track_offsets=False)
    assert len(enc) > 0
    assert enc.offsets is None


def test_encode_rich_repr(bpe_tokenizer):
    enc = bpe_tokenizer.encode_rich("Hello")
    r = repr(enc)
    assert "n_tokens=" in r
