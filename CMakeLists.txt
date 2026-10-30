cmake_minimum_required(VERSION 3.25)
project(skin-tokens.cpp VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SKINTOKENS_BUILD_TESTS "Build tests" ON)
option(SKINTOKENS_ENABLE_VULKAN "Build GGML Vulkan backend" ON)
option(SKINTOKENS_DYNAMIC_BACKENDS "Load CPU/Vulkan backends dynamically" ON)
option(SKINTOKENS_CPU_ALL_VARIANTS "Build portable CPU backend variants" ON)
option(SKINTOKENS_SANITIZE "Enable ASan and UBSan" OFF)
option(SKINTOKENS_FUZZ "Build parser fuzzers" OFF)
option(SKINTOKENS_APPLY_GGML_PATCHES "Apply bundled patches to an isolated GGML build copy" ON)
set(SKINTOKENS_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "Pinned GGML submodule")
set(SKINTOKENS_GGML_PATCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml" CACHE PATH "Ordered GGML patches")
set(SKINTOKENS_GGML_BASE_REVISION "8c63e70982c95ceb862e3a1073a2c1beef75d60a" CACHE STRING "GGML revision required by bundled patches")
set(SKINTOKENS_PARITY_MODEL_DIR "" CACHE PATH "GGUF bundle for opt-in full binding parity tests")
set(SKINTOKENS_PARITY_FIXTURE_DIR "" CACHE PATH "Captured final upstream binding fixture")
set(SKINTOKENS_QWEN_FIXTURE_DIR "" CACHE PATH "Captured TokenRig generation fixture")
set(SKINTOKENS_GIRAFFE_FIXTURE_DIR "" CACHE PATH "Captured official upstream giraffe fixture")

if(SKINTOKENS_SANITIZE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "SKINTOKENS_SANITIZE requires Clang")
  endif()
  add_library(skintokens_sanitizers INTERFACE)
  target_compile_options(skintokens_sanitizers INTERFACE -fno-omit-frame-pointer -fsanitize=address,undefined)
  target_link_options(skintokens_sanitizers INTERFACE -fno-omit-frame-pointer -fsanitize=address,undefined)
endif()

if(NOT EXISTS "${SKINTOKENS_GGML_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Initialize the ggml git submodule or set SKINTOKENS_GGML_SOURCE_DIR")
endif()
set(SKINTOKENS_GGML_EFFECTIVE_SOURCE_DIR "${SKINTOKENS_GGML_SOURCE_DIR}")
if(SKINTOKENS_APPLY_GGML_PATCHES)
  include(cmake/PrepareGGML.cmake)
  skintokens_prepare_ggml(SKINTOKENS_GGML_EFFECTIVE_SOURCE_DIR
    SOURCE_DIR "${SKINTOKENS_GGML_SOURCE_DIR}"
    PATCH_DIR "${SKINTOKENS_GGML_PATCH_DIR}"
    OUTPUT_DIR "${CMAKE_BINARY_DIR}/_deps/ggml-patched-src"
    EXPECTED_REVISION "${SKINTOKENS_GGML_BASE_REVISION}")
else()
  message(WARNING "Building GGML without the bundled Vulkan correctness patches")
endif()
# GGML builds its shader generator through a nested CMake project. If the
# effective source changes (for example when upgrading from the historical
# in-place submodule build), discard only GGML's binary subtree so its nested
# cache cannot retain the previous source path.
set(SKINTOKENS_GGML_SOURCE_MARKER "${CMAKE_BINARY_DIR}/.skintokens-ggml-source")
set(SKINTOKENS_RESET_GGML_BINARY_DIR FALSE)
if(EXISTS "${SKINTOKENS_GGML_SOURCE_MARKER}")
  file(READ "${SKINTOKENS_GGML_SOURCE_MARKER}" SKINTOKENS_PREVIOUS_GGML_SOURCE)
  string(STRIP "${SKINTOKENS_PREVIOUS_GGML_SOURCE}" SKINTOKENS_PREVIOUS_GGML_SOURCE)
  if(NOT SKINTOKENS_PREVIOUS_GGML_SOURCE STREQUAL SKINTOKENS_GGML_EFFECTIVE_SOURCE_DIR)
    set(SKINTOKENS_RESET_GGML_BINARY_DIR TRUE)
  endif()
elseif(EXISTS "${CMAKE_BINARY_DIR}/ggml")
  set(SKINTOKENS_RESET_GGML_BINARY_DIR TRUE)
endif()
if(SKINTOKENS_RESET_GGML_BINARY_DIR)
  message(STATUS "Resetting GGML build metadata for the prepared source tree")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/ggml")
endif()
file(WRITE "${SKINTOKENS_GGML_SOURCE_MARKER}"
  "${SKINTOKENS_GGML_EFFECTIVE_SOURCE_DIR}\n")
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ${SKINTOKENS_ENABLE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ${SKINTOKENS_DYNAMIC_BACKENDS} CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ${SKINTOKENS_CPU_ALL_VARIANTS} CACHE BOOL "" FORCE)
if(SKINTOKENS_DYNAMIC_BACKENDS)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
endif()
add_subdirectory("${SKINTOKENS_GGML_EFFECTIVE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
if(TARGET ggml-vulkan)
  # GGML finds this package in a child scope but does not consume its headers.
  # Import it here too (using the cached package location) for this link call.
  find_package(SPIRV-Headers CONFIG REQUIRED)
  target_link_libraries(ggml-vulkan PRIVATE SPIRV-Headers::SPIRV-Headers)
endif()

# GGML records its public headers as paths relative to the submodule root.
# This parent project owns the install rule (the subdirectory is excluded from
# the default install), so make those paths absolute before CMake evaluates
# them from this directory.
get_target_property(SKINTOKENS_GGML_PUBLIC_HEADERS ggml PUBLIC_HEADER)
set(SKINTOKENS_GGML_PUBLIC_HEADERS_ABSOLUTE)
foreach(header IN LISTS SKINTOKENS_GGML_PUBLIC_HEADERS)
  if(IS_ABSOLUTE "${header}")
    list(APPEND SKINTOKENS_GGML_PUBLIC_HEADERS_ABSOLUTE "${header}")
  else()
    list(APPEND SKINTOKENS_GGML_PUBLIC_HEADERS_ABSOLUTE
      "${SKINTOKENS_GGML_EFFECTIVE_SOURCE_DIR}/${header}")
  endif()
endforeach()
set_target_properties(ggml PROPERTIES
  PUBLIC_HEADER "${SKINTOKENS_GGML_PUBLIC_HEADERS_ABSOLUTE}")

find_package(Threads REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)

add_library(skintokens SHARED
  src/backend.cpp
  src/binding.cpp
  src/c_api.cpp
  src/gguf.cpp
  src/glb.cpp
  src/mesh_encoder.cpp
  src/model.cpp
  src/qwen.cpp
  src/retarget.cpp
  src/skinning.cpp
  src/skin_vae.cpp
  src/tokenizer.cpp
  src/weights.cpp
)
add_library(skintokens::skintokens ALIAS skintokens)
target_include_directories(skintokens PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> PRIVATE src)
target_compile_definitions(skintokens PRIVATE SKINTOKENS_BUILD SKINTOKENS_SHARED)
target_compile_definitions(skintokens PUBLIC SKINTOKENS_SHARED)
target_link_libraries(skintokens PRIVATE ggml Threads::Threads nlohmann_json::nlohmann_json)
if(TARGET skintokens_sanitizers)
  target_link_libraries(skintokens PUBLIC skintokens_sanitizers)
endif()
set_target_properties(skintokens PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
if(MSVC)
  target_compile_options(skintokens PRIVATE /W4 /permissive- /sdl)
else()
  target_compile_options(skintokens PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wformat=2)
endif()

add_executable(skintokens-cli src/cli.cpp)
target_link_libraries(skintokens-cli PRIVATE skintokens::skintokens)
set_target_properties(skintokens-cli PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS skintokens skintokens-cli
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/skintokens DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# add_subdirectory(... EXCLUDE_FROM_ALL) keeps GGML's own install rules out of
# this project. Ship the core runtime and dynamic backends explicitly so a
# normal skin-tokens.cpp install does not depend on its build tree.
install(TARGETS ggml ggml-base
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
foreach(backend IN LISTS GGML_AVAILABLE_BACKENDS)
  if(TARGET ${backend})
    install(TARGETS ${backend}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(UNIX AND NOT APPLE)
      set_property(TARGET ${backend} PROPERTY INSTALL_RPATH
        "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
  endif()
endforeach()
if(UNIX AND NOT APPLE)
  set_property(TARGET ggml ggml-base skintokens PROPERTY INSTALL_RPATH "$ORIGIN")
  set_property(TARGET skintokens-cli PROPERTY INSTALL_RPATH
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

add_executable(skintokens-qwen-parity
  tools/qwen_parity.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-qwen-parity PRIVATE src include)
target_link_libraries(skintokens-qwen-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-qwen-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-mesh-parity
  tools/mesh_parity.cpp src/backend.cpp src/mesh_encoder.cpp src/weights.cpp)
target_include_directories(skintokens-mesh-parity PRIVATE src include)
target_link_libraries(skintokens-mesh-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-mesh-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-vae-parity
  tools/vae_parity.cpp src/backend.cpp src/skin_vae.cpp src/weights.cpp)
target_include_directories(skintokens-vae-parity PRIVATE src include)
target_link_libraries(skintokens-vae-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-vae-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-generation-parity
  tools/generation_parity.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-generation-parity PRIVATE src include)
target_link_libraries(skintokens-generation-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-generation-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-animation-parity tools/animation_parity.cpp)
target_link_libraries(skintokens-animation-parity PRIVATE skintokens::skintokens nlohmann_json::nlohmann_json)
set_target_properties(skintokens-animation-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-full-binding-parity
  tools/full_binding_parity.cpp src/backend.cpp src/binding.cpp src/skinning.cpp src/skin_vae.cpp src/weights.cpp)
target_include_directories(skintokens-full-binding-parity PRIVATE src include)
target_link_libraries(skintokens-full-binding-parity PRIVATE ggml Threads::Threads nlohmann_json::nlohmann_json)
set_target_properties(skintokens-full-binding-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-giraffe-parity
  tools/giraffe_parity.cpp src/backend.cpp src/binding.cpp src/skin_vae.cpp src/weights.cpp)
target_include_directories(skintokens-giraffe-parity PRIVATE src include)
target_link_libraries(skintokens-giraffe-parity PRIVATE ggml Threads::Threads nlohmann_json::nlohmann_json)
set_target_properties(skintokens-giraffe-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-logits-dump
  tools/logits_dump.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-logits-dump PRIVATE src include)
target_link_libraries(skintokens-logits-dump PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-logits-dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(skintokens-qwen-batch-parity
  tools/qwen_batch_parity.cpp src/backend.cpp src/qwen.cpp src/weights.cpp)
target_include_directories(skintokens-qwen-batch-parity PRIVATE src include)
target_link_libraries(skintokens-qwen-batch-parity PRIVATE ggml Threads::Threads)
set_target_properties(skintokens-qwen-batch-parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include(CTest)
if(SKINTOKENS_BUILD_TESTS)
  add_subdirectory(tests)
  if(SKINTOKENS_PARITY_MODEL_DIR AND SKINTOKENS_PARITY_FIXTURE_DIR)
    add_test(NAME skintokens-final-binding-parity-cpu COMMAND
      skintokens-full-binding-parity
      "${SKINTOKENS_PARITY_MODEL_DIR}" "${SKINTOKENS_PARITY_FIXTURE_DIR}" cpu)
    set_tests_properties(skintokens-final-binding-parity-cpu PROPERTIES
      LABELS "model;parity" TIMEOUT 900)
    if(SKINTOKENS_ENABLE_VULKAN)
      add_test(NAME skintokens-final-binding-parity-vulkan COMMAND
        skintokens-full-binding-parity
        "${SKINTOKENS_PARITY_MODEL_DIR}" "${SKINTOKENS_PARITY_FIXTURE_DIR}" vulkan)
      set_tests_properties(skintokens-final-binding-parity-vulkan PROPERTIES
        LABELS "model;parity;gpu" TIMEOUT 900)
    endif()
  endif()
  if(SKINTOKENS_PARITY_MODEL_DIR AND SKINTOKENS_QWEN_FIXTURE_DIR)
    add_test(NAME skintokens-qwen-batch-parity-cpu COMMAND
      skintokens-qwen-batch-parity
      "${SKINTOKENS_PARITY_MODEL_DIR}" "${SKINTOKENS_QWEN_FIXTURE_DIR}" cpu)
    set_tests_properties(skintokens-qwen-batch-parity-cpu PROPERTIES
      LABELS "model;parity;batching" TIMEOUT 900)
    if(SKINTOKENS_ENABLE_VULKAN)
      add_test(NAME skintokens-qwen-batch-parity-vulkan COMMAND
        skintokens-qwen-batch-parity
        "${SKINTOKENS_PARITY_MODEL_DIR}" "${SKINTOKENS_QWEN_FIXTURE_DIR}" vulkan)
      set_tests_properties(skintokens-qwen-batch-parity-vulkan PROPERTIES
        LABELS "model;parity;batching;gpu" TIMEOUT 900)
    endif()
  endif()
  if(SKINTOKENS_PARITY_MODEL_DIR AND SKINTOKENS_GIRAFFE_FIXTURE_DIR)
    add_test(NAME skintokens-giraffe-parity-cpu COMMAND
      skintokens-giraffe-parity
      "${SKINTOKENS_PARITY_MODEL_DIR}" "${SKINTOKENS_GIRAFFE_FIXTURE_DIR}" cpu)
    set_tests_properties(skintokens-giraffe-parity-cpu PROPERTIES
      LABELS "model;parity;official-example" TIMEOUT 1800)
    if(SKINTOKENS_ENABLE_VULKAN)
      add_test(NAME skintokens-giraffe-parity-vulkan COMMAND
        skintokens-giraffe-parity
        "${SKINTOKENS_PARITY_MODEL_DIR}" "${SKINTOKENS_GIRAFFE_FIXTURE_DIR}" vulkan)
      set_tests_properties(skintokens-giraffe-parity-vulkan PROPERTIES
        LABELS "model;parity;official-example;gpu" TIMEOUT 1800)
    endif()
  endif()
endif()
if(SKINTOKENS_FUZZ)
  add_subdirectory(fuzz)
endif()
