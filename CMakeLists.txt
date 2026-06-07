cmake_minimum_required(VERSION 3.18)
project(dots_tts_cpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ---- ggml library (from llama.cpp) ----
# Build ggml as a subdirectory — inherits its CMakeLists.txt
set(GGML_BUILD_COMMIT "dots_tts_cpp")
set(GGML_BUILD_NUMBER 0)
add_subdirectory(${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml ggml_build)

# ---- dots.tts.cpp ----
set(DOTS_TTS_SOURCES
    src/modules/backbone/dit_forward.cpp
    src/modules/backbone/flow_matching.cpp
    src/modules/backbone/patchenc.cpp
    src/main.cpp
)

add_executable(dots_tts ${DOTS_TTS_SOURCES})

target_include_directories(dots_tts PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../llama/llama.cpp/ggml/src
)

target_link_libraries(dots_tts PRIVATE ggml pthread m)

# Optional: OpenMP for parallelism
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dots_tts PRIVATE OpenMP::OpenMP_CXX)
endif()
