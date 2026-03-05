# iree-tokenizer-pythonr

Python bindings for the [IREE](https://github.com/iree-org/iree) tokenizer —
a high-performance C tokenizer with full HuggingFace `tokenizer.json`
compatibility.

- **Fast.** 3–12x faster than tiktoken, 10–20x faster than HF tokenizers.
  Pure C hot path with zero allocations per token.
- **Zero Python dependencies** beyond numpy.
- **Small.** ~317KiB (compared to 1-3MiB for alternatives).
- **Streaming encode/decode.** First-class support for incremental
  tokenization — feed chunks in, get tokens out. Ideal for LLM inference.
- **Drop-in compatible.** Loads any HuggingFace `tokenizer.json`. Supports
  BPE, WordPiece, and Unigram models.

Based on the [IREE high-speed tokenizer library](https://github.com/iree-org/iree/blob/main/runtime/src/iree/tokenizer/README.md):

- **Optimized for cache utilization.** Efficiently utilizes cache on both large and small CPUs. No dependencies and small footprint make it ideal for embedded/client and inclusion into other projects.
- **Unique Algorithmic optimizations.** Pull-based streaming processor with bounded/small, deterministic memory usage. Various novel optimizations not seen elsewhere.
- **GPU-ready.** Designed to be compatible with executing tiled on the GPU,
  not just the host.

## Performance

GPT-2 tokenizer, single-threaded, p50 latency over 50 iterations.

```
Encode (22K chars → 5000 tokens)
  iree       469 µs    10.6M tok/s
  tiktoken  1251 µs     4.0M tok/s   2.7x slower
  hf        5420 µs     0.9M tok/s  11.6x slower

Decode (5000 tokens → text)
  iree        72 µs
  tiktoken    78 µs                   1.1x slower
  hf         599 µs                   8.3x slower

Batch Encode (100 × 880 chars)
  iree      1942 µs    10.3M tok/s
  tiktoken  5148 µs     3.8M tok/s   2.7x slower
  hf       22022 µs     0.9M tok/s  11.3x slower
```

Measured on AMD Threadripper 3970X, 128 GB DDR4, Fedora 43, GCC 15.2, Python 3.14.

## Quick Start

```python
from iree.tokenizer import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

# Encode / decode
ids = tok.encode("Hello world")          # [15496, 995]
text = tok.decode(ids)                    # "Hello world"

# Batch
tok.encode_batch(["Hello", "world"])      # [[15496], [995]]

# Numpy (zero-copy)
arr = tok.encode_to_array("Hello world")  # int32 ndarray

# Rich encoding with byte offsets
enc = tok.encode_rich("Hello world", track_offsets=True)
# enc.ids, enc.offsets, enc.type_ids

# Streaming decode (LLM token-at-a-time pattern)
from iree.tokenizer import decode_stream_iter
for chunk in decode_stream_iter(tok, token_generator):
    print(chunk, end="", flush=True)
```

## API

| Method | Returns | Description |
|--------|---------|-------------|
| `Tokenizer.from_file(path)` | `Tokenizer` | Load from `tokenizer.json` |
| `Tokenizer.from_str(json)` | `Tokenizer` | Load from JSON string |
| `Tokenizer.from_buffer(bytes)` | `Tokenizer` | Load from bytes |
| `tok.encode(text)` | `list[int]` | Encode text to token IDs |
| `tok.encode_to_array(text)` | `np.ndarray` | Encode to numpy int32 array |
| `tok.encode_rich(text)` | `Encoding` | IDs + byte offsets + type IDs |
| `tok.decode(ids)` | `str` | Decode token IDs to text |
| `tok.encode_batch(texts)` | `list[list[int]]` | Batch encode |
| `tok.decode_batch(id_lists)` | `list[str]` | Batch decode |
| `tok.encode_stream()` | `EncodeStream` | Streaming encoder (context manager) |
| `tok.decode_stream()` | `DecodeStream` | Streaming decoder (context manager) |
| `tok.vocab_size` | `int` | Vocabulary size |
| `tok.model_type` | `str` | `"BPE"`, `"WordPiece"`, or `"Unigram"` |
| `tok.token_to_id(token)` | `int \| None` | Look up token ID |
| `tok.id_to_token(id)` | `str \| None` | Look up token text |

## CLI

A streaming `iree-tokenizer-python` command is included. It reads from stdin, writes
JSONL to stdout, and shows live throughput on stderr.

```bash
# Encode text to token IDs
echo "Hello world" | iree-tokenizer-python encode -t tokenizer.json
# {"seq":0,"text":"Hello world","ids":[15496,995],"n_tokens":2,...}

# Decode token IDs back to text
echo '[15496, 995]' | iree-tokenizer-python decode -t tokenizer.json
# {"seq":0,"ids":[15496,995],"text":"Hello world","n_tokens":2,...}

# Chain encode → decode (round-trip)
cat corpus.txt | iree-tokenizer-python encode -t tokenizer.json | iree-tokenizer-python decode -t tokenizer.json

# Tokenizer info
iree-tokenizer-python info -t tokenizer.json
```

Output is chainable: encode output feeds directly into decode and vice versa.
Use `--compact` to omit timing fields, `--rich` for byte offsets, or
`--no-progress` to suppress the stderr throughput display.

Note that this tool illustrates streaming processing but the overhead of JSON
processing is expensive and skews throughput. Treat this as an example of how
to operate the streaming API vs a benchmarking tool or a tool expected to
achieve maximum throughput.

## License

Apache 2.0 with LLVM Exceptions — see [LICENSE](LICENSE).
