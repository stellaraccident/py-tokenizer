# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import pytest

from iree.tokenizer import Tokenizer

TESTDATA = pathlib.Path(__file__).parent / "testdata"


@pytest.fixture
def bpe_tokenizer():
    """Minimal BPE byte-level tokenizer for testing."""
    return Tokenizer.from_file(str(TESTDATA / "bpe_bytelevel_minimal.json"))


@pytest.fixture
def tiktoken_tokenizer():
    """Minimal tiktoken (gpt2 encoding) tokenizer for testing."""
    return Tokenizer.from_tiktoken(
        str(TESTDATA / "tiktoken_gpt2.tiktoken"), encoding="gpt2"
    )
