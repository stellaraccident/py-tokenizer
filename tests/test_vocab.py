# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from iree.tokenizer import Tokenizer


def test_vocab_size(bpe_tokenizer):
    assert bpe_tokenizer.vocab_size == 112


def test_model_type(bpe_tokenizer):
    assert bpe_tokenizer.model_type == "BPE"


def test_token_to_id(bpe_tokenizer):
    result = bpe_tokenizer.token_to_id("H")
    assert result == 39


def test_token_to_id_not_found(bpe_tokenizer):
    result = bpe_tokenizer.token_to_id("\x00\x01\x02\x03nonexistent")
    assert result is None


def test_id_to_token(bpe_tokenizer):
    token = bpe_tokenizer.id_to_token(0)
    assert token == "!"


def test_id_to_token_out_of_range(bpe_tokenizer):
    result = bpe_tokenizer.id_to_token(999999)
    assert result is None


def test_id_to_token_negative(bpe_tokenizer):
    result = bpe_tokenizer.id_to_token(-1)
    assert result is None


def test_roundtrip_token_id(bpe_tokenizer):
    token = bpe_tokenizer.id_to_token(0)
    assert token is not None
    id_back = bpe_tokenizer.token_to_id(token)
    assert id_back == 0
