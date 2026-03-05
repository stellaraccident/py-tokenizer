// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/tokenizer.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "encoding.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/format/tiktoken/tiktoken.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "status_util.h"

namespace nb = nanobind;
using namespace iree::tokenizer::python;

#include "streaming.h"

// ---------------------------------------------------------------------------
// TokenizerWrapper — prevents double-free via unique_ptr.
// ---------------------------------------------------------------------------

struct TokenizerDeleter {
  void operator()(iree_tokenizer_t* t) {
    if (t) iree_tokenizer_free(t);
  }
};
using TokenizerPtr = std::unique_ptr<iree_tokenizer_t, TokenizerDeleter>;

class TokenizerWrapper {
 public:
  explicit TokenizerWrapper(iree_tokenizer_t* raw) : ptr_(raw) {}

  const iree_tokenizer_t* get() const { return ptr_.get(); }

  // -------------------------------------------------------------------------
  // Encode
  // -------------------------------------------------------------------------

  std::vector<int32_t> Encode(const std::string& text,
                              bool add_special_tokens) {
    iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    if (add_special_tokens) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
    }

    // Estimate output size: ~1 token per 4 bytes + room for special tokens.
    size_t capacity = text.size() / 2 + 16;
    std::vector<iree_tokenizer_token_id_t> token_ids(capacity);
    iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
        token_ids.data(), nullptr, nullptr, capacity);
    iree_host_size_t token_count = 0;

    iree_string_view_t sv = {.data = text.data(), .size = text.size()};
    iree_status_t status = iree_tokenizer_encode(
        ptr_.get(), sv, flags, output, iree_allocator_system(), &token_count);

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_free(status);
      // Retry with larger buffer.
      capacity = text.size() + 64;
      token_ids.resize(capacity);
      output = iree_tokenizer_make_token_output(token_ids.data(), nullptr,
                                                nullptr, capacity);
      status = iree_tokenizer_encode(ptr_.get(), sv, flags, output,
                                     iree_allocator_system(), &token_count);
    }
    CheckStatus(status);
    token_ids.resize(token_count);
    return token_ids;
  }

  nb::ndarray<nb::numpy, int32_t, nb::ndim<1>> EncodeToArray(
      const std::string& text, bool add_special_tokens) {
    auto ids = Encode(text, add_special_tokens);
    size_t n = ids.size();
    auto data = std::make_unique<int32_t[]>(n);
    std::memcpy(data.get(), ids.data(), n * sizeof(int32_t));
    int32_t* raw = data.release();
    nb::capsule owner(raw, [](void* p) noexcept { delete[] (int32_t*)p; });
    return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(raw, {n},
                                                        std::move(owner));
  }

  // -------------------------------------------------------------------------
  // Decode
  // -------------------------------------------------------------------------

  std::string Decode(const std::vector<int32_t>& tokens,
                     bool skip_special_tokens) {
    iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE;
    if (skip_special_tokens) {
      flags |= IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
    }

    iree_tokenizer_token_id_list_t list = {
        .count = tokens.size(),
        .values = tokens.data(),
    };

    // Estimate output size: ~4 bytes per token.
    size_t capacity = tokens.size() * 4 + 64;
    std::vector<char> text_buf(capacity);
    iree_mutable_string_view_t text_output = {.data = text_buf.data(),
                                              .size = text_buf.size()};
    iree_host_size_t text_length = 0;

    iree_status_t status =
        iree_tokenizer_decode(ptr_.get(), list, flags, text_output,
                              iree_allocator_system(), &text_length);

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_free(status);
      // Retry with 2x buffer.
      capacity *= 2;
      text_buf.resize(capacity);
      text_output = {.data = text_buf.data(), .size = text_buf.size()};
      status = iree_tokenizer_decode(ptr_.get(), list, flags, text_output,
                                     iree_allocator_system(), &text_length);
    }
    CheckStatus(status);
    return std::string(text_buf.data(), text_length);
  }

  // -------------------------------------------------------------------------
  // Batch Encode
  // -------------------------------------------------------------------------

  std::vector<std::vector<int32_t>> EncodeBatch(
      const std::vector<std::string>& texts, bool add_special_tokens) {
    if (texts.empty()) return {};

    iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    if (add_special_tokens) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
    }

    // Allocate per-item token buffers and batch items.
    size_t item_count = texts.size();
    std::vector<std::vector<iree_tokenizer_token_id_t>> token_bufs(item_count);
    std::vector<iree_tokenizer_encode_batch_item_t> items(item_count);

    for (size_t i = 0; i < item_count; ++i) {
      size_t cap = texts[i].size() / 2 + 16;
      token_bufs[i].resize(cap);
      items[i].text = {.data = texts[i].data(), .size = texts[i].size()};
      items[i].output = iree_tokenizer_make_token_output(token_bufs[i].data(),
                                                         nullptr, nullptr, cap);
      items[i].out_token_count = 0;
    }

    // Allocate shared state.
    iree_host_size_t state_size = 0;
    CheckStatus(
        iree_tokenizer_encode_state_calculate_size(ptr_.get(), &state_size));
    std::vector<uint8_t> state_storage(state_size);

    // Find max text for transform buffer sizing.
    size_t max_text = 0;
    for (const auto& t : texts) max_text = std::max(max_text, t.size());
    iree_host_size_t tb_size =
        iree_tokenizer_transform_buffer_oneshot_size(max_text);
    std::vector<uint8_t> transform_buffer(tb_size);

    iree_status_t status = iree_tokenizer_encode_batch(
        ptr_.get(), items.data(), item_count, flags,
        iree_make_byte_span(state_storage.data(), state_storage.size()),
        iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
        iree_tokenizer_offset_run_list_empty());

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_free(status);
      // Retry with larger per-item buffers.
      for (size_t i = 0; i < item_count; ++i) {
        size_t cap = texts[i].size() + 64;
        token_bufs[i].resize(cap);
        items[i].output = iree_tokenizer_make_token_output(
            token_bufs[i].data(), nullptr, nullptr, cap);
        items[i].out_token_count = 0;
      }
      status = iree_tokenizer_encode_batch(
          ptr_.get(), items.data(), item_count, flags,
          iree_make_byte_span(state_storage.data(), state_storage.size()),
          iree_make_byte_span(transform_buffer.data(), transform_buffer.size()),
          iree_tokenizer_offset_run_list_empty());
    }
    CheckStatus(status);

    std::vector<std::vector<int32_t>> result(item_count);
    for (size_t i = 0; i < item_count; ++i) {
      result[i].assign(token_bufs[i].data(),
                       token_bufs[i].data() + items[i].out_token_count);
    }
    return result;
  }

  // -------------------------------------------------------------------------
  // Batch Encode to Array (flat + lengths)
  // -------------------------------------------------------------------------

  nb::tuple EncodeBatchToArray(const std::vector<std::string>& texts,
                               bool add_special_tokens) {
    auto batched = EncodeBatch(texts, add_special_tokens);
    // Compute total size and per-item lengths.
    size_t total = 0;
    for (const auto& ids : batched) total += ids.size();

    auto flat_ptr = std::make_unique<int32_t[]>(total);
    auto len_ptr = std::make_unique<int64_t[]>(batched.size());
    size_t offset = 0;
    for (size_t i = 0; i < batched.size(); ++i) {
      std::memcpy(flat_ptr.get() + offset, batched[i].data(),
                  batched[i].size() * sizeof(int32_t));
      len_ptr[i] = static_cast<int64_t>(batched[i].size());
      offset += batched[i].size();
    }

    int32_t* flat_data = flat_ptr.release();
    int64_t* len_data = len_ptr.release();
    nb::capsule flat_owner(flat_data,
                           [](void* p) noexcept { delete[] (int32_t*)p; });
    nb::capsule len_owner(len_data,
                          [](void* p) noexcept { delete[] (int64_t*)p; });

    auto flat_arr = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(
        flat_data, {total}, std::move(flat_owner));
    auto len_arr = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>(
        len_data, {batched.size()}, std::move(len_owner));

    return nb::make_tuple(std::move(flat_arr), std::move(len_arr));
  }

  // -------------------------------------------------------------------------
  // Batch Decode
  // -------------------------------------------------------------------------

  std::vector<std::string> DecodeBatch(
      const std::vector<std::vector<int32_t>>& token_lists,
      bool skip_special_tokens) {
    if (token_lists.empty()) return {};

    iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE;
    if (skip_special_tokens) {
      flags |= IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
    }

    size_t item_count = token_lists.size();
    std::vector<std::vector<char>> text_bufs(item_count);
    std::vector<iree_tokenizer_decode_batch_item_t> items(item_count);

    for (size_t i = 0; i < item_count; ++i) {
      size_t cap = token_lists[i].size() * 4 + 64;
      text_bufs[i].resize(cap);
      items[i].tokens = {.count = token_lists[i].size(),
                         .values = token_lists[i].data()};
      items[i].text_output = {.data = text_bufs[i].data(),
                              .size = text_bufs[i].size()};
      items[i].out_text_length = 0;
    }

    iree_host_size_t state_size = 0;
    CheckStatus(
        iree_tokenizer_decode_state_calculate_size(ptr_.get(), &state_size));
    std::vector<uint8_t> state_storage(state_size);

    iree_status_t status = iree_tokenizer_decode_batch(
        ptr_.get(), items.data(), item_count, flags,
        iree_make_byte_span(state_storage.data(), state_storage.size()));

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_free(status);
      for (size_t i = 0; i < item_count; ++i) {
        size_t cap = text_bufs[i].size() * 2;
        text_bufs[i].resize(cap);
        items[i].text_output = {.data = text_bufs[i].data(),
                                .size = text_bufs[i].size()};
        items[i].out_text_length = 0;
      }
      status = iree_tokenizer_decode_batch(
          ptr_.get(), items.data(), item_count, flags,
          iree_make_byte_span(state_storage.data(), state_storage.size()));
    }
    CheckStatus(status);

    std::vector<std::string> result(item_count);
    for (size_t i = 0; i < item_count; ++i) {
      result[i].assign(text_bufs[i].data(), items[i].out_text_length);
    }
    return result;
  }

  // -------------------------------------------------------------------------
  // Rich Encode
  // -------------------------------------------------------------------------

  nb::object EncodeRich(const std::string& text, bool add_special_tokens,
                        bool track_offsets) {
    iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    if (add_special_tokens) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
    }

    size_t capacity = text.size() / 2 + 16;
    std::vector<iree_tokenizer_token_id_t> token_ids(capacity);
    std::vector<iree_tokenizer_offset_t> offsets;
    std::vector<uint8_t> type_ids(capacity, 0);

    iree_tokenizer_offset_t* offset_ptr = nullptr;
    if (track_offsets) {
      offsets.resize(capacity);
      offset_ptr = offsets.data();
      flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
    }

    iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
        token_ids.data(), offset_ptr, type_ids.data(), capacity);
    iree_host_size_t token_count = 0;

    iree_string_view_t sv = {.data = text.data(), .size = text.size()};
    iree_status_t status = iree_tokenizer_encode(
        ptr_.get(), sv, flags, output, iree_allocator_system(), &token_count);

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_free(status);
      capacity = text.size() + 64;
      token_ids.resize(capacity);
      type_ids.resize(capacity, 0);
      if (track_offsets) {
        offsets.resize(capacity);
        offset_ptr = offsets.data();
      }
      output = iree_tokenizer_make_token_output(token_ids.data(), offset_ptr,
                                                type_ids.data(), capacity);
      status = iree_tokenizer_encode(ptr_.get(), sv, flags, output,
                                     iree_allocator_system(), &token_count);
    }
    CheckStatus(status);

    // Build Encoding object.
    size_t n = token_count;

    // IDs array.
    auto ids_ptr = std::make_unique<int32_t[]>(n);
    std::memcpy(ids_ptr.get(), token_ids.data(), n * sizeof(int32_t));
    int32_t* ids_data = ids_ptr.release();
    nb::capsule ids_owner(ids_data,
                          [](void* p) noexcept { delete[] (int32_t*)p; });
    auto ids_arr = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(
        ids_data, {n}, std::move(ids_owner));

    // Offsets array — (n, 2) of uint64.
    nb::object offsets_arr;
    if (track_offsets) {
      auto off_ptr = std::make_unique<uint64_t[]>(n * 2);
      for (size_t i = 0; i < n; ++i) {
        off_ptr[i * 2] = offsets[i].start;
        off_ptr[i * 2 + 1] = offsets[i].end;
      }
      uint64_t* off_data = off_ptr.release();
      nb::capsule off_owner(off_data,
                            [](void* p) noexcept { delete[] (uint64_t*)p; });
      offsets_arr = nb::cast(nb::ndarray<nb::numpy, uint64_t, nb::ndim<2>>(
          off_data, {n, 2}, std::move(off_owner)));
    } else {
      offsets_arr = nb::none();
    }

    // Type IDs array.
    auto tid_ptr = std::make_unique<uint8_t[]>(n);
    std::memcpy(tid_ptr.get(), type_ids.data(), n);
    uint8_t* tid_data = tid_ptr.release();
    nb::capsule tid_owner(tid_data,
                          [](void* p) noexcept { delete[] (uint8_t*)p; });
    auto tid_arr = nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(
        tid_data, {n}, std::move(tid_owner));

    Encoding enc;
    enc.ids = nb::cast(std::move(ids_arr));
    enc.offsets = std::move(offsets_arr);
    enc.type_ids = nb::cast(std::move(tid_arr));
    return nb::cast(std::move(enc));
  }

  // -------------------------------------------------------------------------
  // Vocabulary
  // -------------------------------------------------------------------------

  size_t vocab_size() const {
    const iree_tokenizer_vocab_t* v = iree_tokenizer_vocab(ptr_.get());
    return iree_tokenizer_vocab_token_count(v);
  }

  std::string model_type() const {
    iree_string_view_t name = iree_tokenizer_model_type_name(ptr_.get());
    return std::string(name.data, name.size);
  }

  std::optional<int32_t> token_to_id(const std::string& token) const {
    const iree_tokenizer_vocab_t* v = iree_tokenizer_vocab(ptr_.get());
    iree_string_view_t sv = {.data = token.data(), .size = token.size()};
    int32_t id = iree_tokenizer_vocab_lookup(v, sv);
    if (id == IREE_TOKENIZER_TOKEN_ID_INVALID) return std::nullopt;
    return id;
  }

  std::optional<std::string> id_to_token(int32_t id) const {
    const iree_tokenizer_vocab_t* v = iree_tokenizer_vocab(ptr_.get());
    if (id < 0 ||
        static_cast<size_t>(id) >= iree_tokenizer_vocab_token_count(v))
      return std::nullopt;
    iree_string_view_t sv = iree_tokenizer_vocab_token_text(v, id);
    return std::string(sv.data, sv.size);
  }

  std::optional<int32_t> special_id(int32_t raw) const {
    return raw == IREE_TOKENIZER_TOKEN_ID_INVALID ? std::nullopt
                                                  : std::optional<int32_t>(raw);
  }

  std::optional<int32_t> bos_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get())).bos);
  }
  std::optional<int32_t> eos_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get())).eos);
  }
  std::optional<int32_t> unk_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get())).unk);
  }
  std::optional<int32_t> pad_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get())).pad);
  }
  std::optional<int32_t> sep_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get())).sep);
  }
  std::optional<int32_t> cls_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get())).cls);
  }
  std::optional<int32_t> mask_token_id() const {
    return special_id(
        iree_tokenizer_vocab_special_ids(iree_tokenizer_vocab(ptr_.get()))
            .mask);
  }

 private:
  TokenizerPtr ptr_;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

static TokenizerWrapper FromJsonString(const std::string& json) {
  iree_tokenizer_t* tokenizer = nullptr;
  iree_string_view_t sv = {.data = json.data(), .size = json.size()};
  CheckStatus(iree_tokenizer_from_huggingface_json(sv, iree_allocator_system(),
                                                   &tokenizer));
  return TokenizerWrapper(tokenizer);
}

static TokenizerWrapper FromTiktokenString(const std::string& data,
                                           const std::string& encoding) {
  iree_string_view_t enc_sv = {.data = encoding.data(),
                               .size = encoding.size()};
  const iree_tokenizer_tiktoken_config_t* config =
      iree_tokenizer_tiktoken_config_by_name(enc_sv);
  if (!config) {
    throw nb::value_error(
        ("Unknown tiktoken encoding: " + encoding +
         ". Supported: cl100k_base, o200k_base, o200k_harmony, "
         "r50k_base, gpt2, p50k_base, p50k_edit")
            .c_str());
  }
  iree_tokenizer_t* tokenizer = nullptr;
  iree_string_view_t data_sv = {.data = data.data(), .size = data.size()};
  CheckStatus(iree_tokenizer_from_tiktoken(
      data_sv, config, iree_allocator_system(), &tokenizer));
  return TokenizerWrapper(tokenizer);
}

static std::string ReadFileContents(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw nb::value_error(("Cannot open file: " + path).c_str());
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  if (size < 0) {
    fclose(f);
    throw nb::value_error("Cannot determine file size (not a regular file?)");
  }
  fseek(f, 0, SEEK_SET);
  std::string contents(size, '\0');
  size_t read = fread(contents.data(), 1, size, f);
  fclose(f);
  if (static_cast<long>(read) != size)
    throw nb::value_error("Failed to read file");
  return contents;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void RegisterTokenizer(nb::module_& m) {
  nb::class_<TokenizerWrapper>(m, "Tokenizer",
                               "High-performance tokenizer backed by IREE.")
      // Construction.
      .def_static(
          "from_file",
          [](const std::string& path) -> TokenizerWrapper {
            return FromJsonString(ReadFileContents(path));
          },
          nb::arg("path"),
          "Load a tokenizer from a HuggingFace tokenizer.json file.")
      .def_static("from_str", &FromJsonString, nb::arg("json"),
                  "Load a tokenizer from a JSON string.")
      .def_static(
          "from_buffer",
          [](nb::bytes data) -> TokenizerWrapper {
            std::string json(data.c_str(), data.size());
            return FromJsonString(json);
          },
          nb::arg("data"), "Load a tokenizer from bytes.")
      .def_static(
          "from_tiktoken",
          [](const std::string& path,
             const std::string& encoding) -> TokenizerWrapper {
            return FromTiktokenString(ReadFileContents(path), encoding);
          },
          nb::arg("path"), nb::arg("encoding"),
          "Load a tokenizer from a .tiktoken file and encoding name.\n\n"
          "Supported encodings: cl100k_base, o200k_base, o200k_harmony, "
          "r50k_base, gpt2, p50k_base, p50k_edit.")
      .def_static("from_tiktoken_str", &FromTiktokenString, nb::arg("data"),
                  nb::arg("encoding"),
                  "Load a tokenizer from tiktoken data string and encoding "
                  "name.")
      .def_static(
          "from_tiktoken_buffer",
          [](nb::bytes data, const std::string& encoding) -> TokenizerWrapper {
            std::string str(data.c_str(), data.size());
            return FromTiktokenString(str, encoding);
          },
          nb::arg("data"), nb::arg("encoding"),
          "Load a tokenizer from tiktoken bytes and encoding name.")

      // Encode.
      .def("encode", &TokenizerWrapper::Encode, nb::arg("text"),
           nb::arg("add_special_tokens") = false, "Encode text to token IDs.")
      .def("encode_to_array", &TokenizerWrapper::EncodeToArray, nb::arg("text"),
           nb::arg("add_special_tokens") = false,
           "Encode text to a numpy int32 array of token IDs.")
      .def("encode_rich", &TokenizerWrapper::EncodeRich, nb::arg("text"),
           nb::arg("add_special_tokens") = false,
           nb::arg("track_offsets") = true,
           "Encode text to an Encoding with IDs, offsets, and type IDs.")

      // Decode.
      .def("decode", &TokenizerWrapper::Decode, nb::arg("tokens"),
           nb::arg("skip_special_tokens") = false, "Decode token IDs to text.")

      // Batch.
      .def("encode_batch", &TokenizerWrapper::EncodeBatch, nb::arg("texts"),
           nb::arg("add_special_tokens") = false,
           "Encode multiple texts to lists of token IDs.")
      .def("encode_batch_to_array", &TokenizerWrapper::EncodeBatchToArray,
           nb::arg("texts"), nb::arg("add_special_tokens") = false,
           "Encode multiple texts to (flat_ids, lengths) numpy arrays.")
      .def("decode_batch", &TokenizerWrapper::DecodeBatch,
           nb::arg("token_lists"), nb::arg("skip_special_tokens") = false,
           "Decode multiple token ID lists to strings.")

      // Streaming.
      .def(
          "encode_stream",
          [](TokenizerWrapper& self, bool add_special_tokens) {
            return new EncodeStream(self.get(), add_special_tokens);
          },
          nb::arg("add_special_tokens") = false, nb::rv_policy::take_ownership,
          nb::keep_alive<0, 1>(),  // Stream ref keeps Tokenizer alive.
          "Create a streaming encoder.")
      .def(
          "decode_stream",
          [](TokenizerWrapper& self, bool skip_special_tokens) {
            return new DecodeStream(self.get(), skip_special_tokens);
          },
          nb::arg("skip_special_tokens") = false, nb::rv_policy::take_ownership,
          nb::keep_alive<0, 1>(), "Create a streaming decoder.")
      // decode_stream_iter is implemented in Python — see __init__.py

      // Vocabulary.
      .def_prop_ro("vocab_size", &TokenizerWrapper::vocab_size)
      .def_prop_ro("model_type", &TokenizerWrapper::model_type)
      .def("token_to_id", &TokenizerWrapper::token_to_id, nb::arg("token"))
      .def("id_to_token", &TokenizerWrapper::id_to_token, nb::arg("id"))
      .def_prop_ro("bos_token_id", &TokenizerWrapper::bos_token_id)
      .def_prop_ro("eos_token_id", &TokenizerWrapper::eos_token_id)
      .def_prop_ro("unk_token_id", &TokenizerWrapper::unk_token_id)
      .def_prop_ro("pad_token_id", &TokenizerWrapper::pad_token_id)
      .def_prop_ro("sep_token_id", &TokenizerWrapper::sep_token_id)
      .def_prop_ro("cls_token_id", &TokenizerWrapper::cls_token_id)
      .def_prop_ro("mask_token_id", &TokenizerWrapper::mask_token_id)

      // Repr.
      .def("__repr__", [](const TokenizerWrapper& self) {
        return "Tokenizer(model_type='" + self.model_type() +
               "', vocab_size=" + std::to_string(self.vocab_size()) + ")";
      });
}
