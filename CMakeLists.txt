# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.24)

# Avoid in-source builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "Do not build in-source. Please remove CMakeCache.txt and the "
    "CMakeFiles/ directory. Then build out-of-source.")
endif()

# Get version number from file.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version.json VERSION_JSON_STRING)
string(JSON PACKAGE_VERSION GET ${VERSION_JSON_STRING} package-version)
string(REGEX MATCH "(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*" BASE_VERSION
  ${PACKAGE_VERSION})

project(
  iree-tokenizer
  VERSION ${BASE_VERSION}
  DESCRIPTION "High-performance Python tokenizer backed by IREE"
  LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)
include(CheckCXXCompilerFlag)

# Options.
option(IREE_TOKENIZER_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(IREE_TOKENIZER_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

################################################################################
# IREE Dependency
#
# Takes a source dependency on the IREE runtime (tokenizer subset).
# To point to a local IREE source directory:
#   -DIREE_SOURCE_DIR=</path/to/iree/source>
#
# When not specified, fetches IREE sources from GitHub at the tag from
# version.json.
################################################################################

string(JSON IREE_VERSION GET ${VERSION_JSON_STRING} iree-version)
set(IREE_GIT_TAG "iree-${IREE_VERSION}")
set(IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# IREE build flags — we only need the tokenizer, not the full runtime.
set(IREE_VISIBILITY_HIDDEN OFF)
set(IREE_BUILD_COMPILER OFF)
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
set(IREE_HAL_DRIVER_DEFAULTS OFF)

# Propagate sanitizer flags to IREE.
if(IREE_TOKENIZER_ENABLE_ASAN)
  set(IREE_ENABLE_ASAN ON)
endif()
if(IREE_TOKENIZER_ENABLE_UBSAN)
  set(IREE_ENABLE_UBSAN ON)
endif()

if(IREE_SOURCE_DIR)
  message(STATUS "Using existing IREE sources: ${IREE_SOURCE_DIR}")
  add_subdirectory(${IREE_SOURCE_DIR} iree_tokenizer_iree SYSTEM
    EXCLUDE_FROM_ALL)
else()
  message(STATUS "Fetching IREE sources from tag ${IREE_GIT_TAG}")
  FetchContent_Declare(
    iree_tokenizer_iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG "${IREE_GIT_TAG}"
    GIT_SUBMODULES "third_party/flatcc"
    GIT_SHALLOW TRUE
    SYSTEM
    EXCLUDE_FROM_ALL
  )
  FetchContent_GetProperties(iree_tokenizer_iree)
  if(NOT iree_tokenizer_iree_POPULATED)
    FetchContent_MakeAvailable(iree_tokenizer_iree)
  endif()
endif()

################################################################################
# nanobind
################################################################################

find_package(Python 3.10 COMPONENTS Interpreter Development.Module
  Development.SABIModule REQUIRED)

# nanobind: prefer system install, fall back to FetchContent.
find_package(nanobind QUIET)
if(NOT nanobind_FOUND)
  message(STATUS "nanobind not found: fetching from GitHub")
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG cc860b2b771a0003ce70415fc56ea18f61b0a153  # v2.9.0
  )
  FetchContent_MakeAvailable(nanobind)
endif()

################################################################################
# Sanitizer Configuration
################################################################################

set(IREE_TOKENIZER_SANITIZER_COMPILE_FLAGS "")
set(IREE_TOKENIZER_SANITIZER_LINK_FLAGS "")

if(IREE_TOKENIZER_ENABLE_ASAN)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "AddressSanitizer requires Clang.")
  endif()
  list(APPEND IREE_TOKENIZER_SANITIZER_COMPILE_FLAGS
    -fsanitize=address -fno-omit-frame-pointer -g)
  list(APPEND IREE_TOKENIZER_SANITIZER_LINK_FLAGS -fsanitize=address)
endif()

if(IREE_TOKENIZER_ENABLE_UBSAN)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "UndefinedBehaviorSanitizer requires Clang.")
  endif()
  list(APPEND IREE_TOKENIZER_SANITIZER_COMPILE_FLAGS
    -fsanitize=undefined -fno-omit-frame-pointer -g -fno-sanitize=vptr
    -fno-sanitize-recover=undefined)
  list(APPEND IREE_TOKENIZER_SANITIZER_LINK_FLAGS -fsanitize=undefined)
endif()

################################################################################
# Python Extension Module
################################################################################

nanobind_add_module(_iree_tokenizer
  NB_STATIC LTO STABLE_ABI FREE_THREADED
  src/bindings/module.cc
  src/bindings/tokenizer.cc
  src/bindings/streaming.cc
  src/bindings/encoding.cc
)

# nanobind requires RTTI and exceptions.
if(MSVC)
  target_compile_options(_iree_tokenizer PRIVATE /EHsc /GR)
else()
  target_compile_options(_iree_tokenizer PRIVATE -frtti -fexceptions)
endif()

# Link IREE tokenizer targets.
target_link_libraries(_iree_tokenizer PRIVATE
  iree_tokenizer_tokenizer
  iree_tokenizer_format_huggingface_tokenizer_json
)

# Apply sanitizer flags.
if(IREE_TOKENIZER_SANITIZER_COMPILE_FLAGS)
  target_compile_options(_iree_tokenizer PRIVATE
    ${IREE_TOKENIZER_SANITIZER_COMPILE_FLAGS})
  target_link_options(_iree_tokenizer PRIVATE
    ${IREE_TOKENIZER_SANITIZER_LINK_FLAGS})
endif()

# Install the native module into the iree/tokenizer package directory.
install(TARGETS _iree_tokenizer
  LIBRARY DESTINATION iree/tokenizer)
