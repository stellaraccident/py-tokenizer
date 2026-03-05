# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from iree.tokenizer import Tokenizer


def test_encode_batch(bpe_tokenizer):
    texts = ["Hello", "world", "foo bar"]
    batch = bpe_tokenizer.encode_batch(texts)
    assert len(batch) == 3
    assert batch[0] == [39, 68, 105]
    assert batch[1] == [86, 108]
    assert batch[2] == [69, 78, 78, 94, 65, 64, 81]


def test_encode_batch_matches_single(bpe_tokenizer):
    texts = ["Hello", "world"]
    batch = bpe_tokenizer.encode_batch(texts)
    for text, batch_ids in zip(texts, batch):
        single_ids = bpe_tokenizer.encode(text)
        assert batch_ids == single_ids


def test_encode_batch_empty(bpe_tokenizer):
    assert bpe_tokenizer.encode_batch([]) == []


def test_decode_batch(bpe_tokenizer):
    texts = ["Hello", "world"]
    batch_ids = bpe_tokenizer.encode_batch(texts)
    decoded = bpe_tokenizer.decode_batch(batch_ids)
    assert decoded == texts


def test_decode_batch_empty(bpe_tokenizer):
    assert bpe_tokenizer.decode_batch([]) == []


def test_encode_batch_to_array(bpe_tokenizer):
    texts = ["Hello", "world"]
    flat, lengths = bpe_tokenizer.encode_batch_to_array(texts)
    assert isinstance(flat, np.ndarray)
    assert flat.dtype == np.int32
    assert isinstance(lengths, np.ndarray)
    assert lengths.dtype == np.int64
    np.testing.assert_array_equal(lengths, [3, 2])
    np.testing.assert_array_equal(flat, [39, 68, 105, 86, 108])
