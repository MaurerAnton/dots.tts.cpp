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

# ---- dots.tts.cpp (legacy target replaced by e2e_pipeline) ----
# dots_tts binary removed — use e2e_pipeline instead

# ---- llama library (pre-built in llama.cpp/build/src) ----

# ---- End-to-end pipeline ----
add_executable(e2e_pipeline
    src/e2e_pipeline.cpp
    src/gpt2_bpe_tokenizer.cpp
    src/gguf_extract.cpp
    src/modules/backbone/dit_forward.cpp
    src/modules/backbone/dit_attention.cpp
    src/modules/backbone/patchenc.cpp
    src/modules/vocoder/audiovae.cpp
    src/modules/vocoder/bigvgan_cpp.cpp
    src/modules/vocoder/lstm.cpp
    src/utils/safetensors.cpp
    src/utils/dit_loader.cpp
    src/utils/patchenc_loader.cpp
    src/modules/speaker/campp.cpp
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
    target_link_libraries(e2e_pipeline PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(e2e_pipeline PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ---- Standalone vocoder test ---- 
add_executable(test_vocoder
    test/test_vocoder.cpp
    src/modules/vocoder/bigvgan_cpp.cpp
    src/modules/vocoder/lstm.cpp
    src/utils/safetensors.cpp
)
target_include_directories(test_vocoder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)
target_link_libraries(test_vocoder PRIVATE ggml pthread m)

# ---- C++ Web demo server (replaces Flask) ----
add_executable(web_server tools/web_server.cpp)
target_link_libraries(web_server PRIVATE pthread)

# ---- Test dots LLM loading ----
add_executable(test_llm test/test_llm.cpp)
target_include_directories(test_llm PRIVATE
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/common
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)
target_link_libraries(test_llm PRIVATE ${LLAMA_LIB} ggml pthread m)

# ---- Test C++ BPE tokenizer ----
add_executable(test_tokenizer test/test_tokenizer.cpp src/gpt2_bpe_tokenizer.cpp)
target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_tokenizer PRIVATE pthread)

# ---- Test CAM++ speaker encoder ----
add_executable(test_campp test/test_campp.cpp src/modules/speaker/campp.cpp src/utils/safetensors.cpp)
target_include_directories(test_campp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)
target_link_libraries(test_campp PRIVATE ggml pthread m)
