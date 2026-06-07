cmake_minimum_required(VERSION 3.18)
project(dots_tts_cpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ---- ggml library (from llama.cpp) ----
set(GGML_BUILD_COMMIT "dots_tts_cpp")
set(GGML_BUILD_NUMBER 0)
add_subdirectory(${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml ggml_build)

# ---- llama library (pre-built in llama.cpp/build/src) ----
set(LLAMA_BUILD_DIR ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/build)
find_library(LLAMA_LIB llama PATHS ${LLAMA_BUILD_DIR}/src NO_DEFAULT_PATH)
if(NOT LLAMA_LIB)
    message(FATAL_ERROR "llama library not found. Build llama.cpp first: cd ../llama/llama.cpp && cmake -B build && cmake --build build -j8 --target llama")
endif()
message(STATUS "Found llama: ${LLAMA_LIB}")

# ---- dots.tts.cpp ----
set(DOTS_TTS_SOURCES
    src/modules/backbone/dit_forward.cpp
    src/modules/backbone/dit_attention.cpp
    src/modules/backbone/flow_matching.cpp
    src/modules/backbone/patchenc.cpp
    src/modules/vocoder/audiovae.cpp
    src/modules/vocoder/bigvgan_cpp.cpp
    src/modules/vocoder/lstm.cpp
    src/utils/safetensors.cpp
    src/utils/dit_loader.cpp
    src/utils/patchenc_loader.cpp
    src/main.cpp
)

add_executable(dots_tts ${DOTS_TTS_SOURCES})

target_include_directories(dots_tts PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)

target_link_libraries(dots_tts PRIVATE ${LLAMA_LIB} ggml pthread m)

# ---- LLM test (minimal, no dummy weights) ----
add_executable(llm_test src/llm_test.cpp)
target_include_directories(llm_test PRIVATE
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)
target_link_libraries(llm_test PRIVATE ${LLAMA_LIB} ggml pthread m)

# ---- End-to-end pipeline ----
add_executable(e2e_pipeline
    src/e2e_pipeline.cpp
    src/modules/backbone/dit_forward.cpp
    src/modules/backbone/dit_attention.cpp
    src/modules/backbone/patchenc.cpp
    src/modules/vocoder/audiovae.cpp
    src/modules/vocoder/bigvgan_cpp.cpp
    src/modules/vocoder/lstm.cpp
    src/utils/safetensors.cpp
    src/utils/dit_loader.cpp
    src/utils/patchenc_loader.cpp
)
target_include_directories(e2e_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)
target_link_libraries(e2e_pipeline PRIVATE ${LLAMA_LIB} ggml pthread m)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dots_tts PRIVATE OpenMP::OpenMP_CXX)
endif()
