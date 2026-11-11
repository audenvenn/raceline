cmake_minimum_required(VERSION 3.24)
project(raceline LANGUAGES C CXX OBJCXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  message(FATAL_ERROR "raceline targets Apple Silicon Macs only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -fobjc-arc")

# Engine core. Metal kernels are compiled at run time from kernels/, so no
# Xcode or offline Metal toolchain is needed to build.
add_library(raceline_core STATIC
  src/metal_rt.mm
  src/model_file.mm
  src/engine.mm
)
target_include_directories(raceline_core PUBLIC src)
target_compile_definitions(raceline_core PUBLIC RL_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels")
target_link_libraries(raceline_core PUBLIC "-framework Metal" "-framework Foundation" "-framework Accelerate")

# Development command line: dump, logits, bench, profile.
add_executable(raceline-engine src/main.mm)
target_link_libraries(raceline-engine PRIVATE raceline_core)

# C API for the Python agent runtime (raceline/engine.py loads it with ctypes).
add_library(raceline SHARED src/capi.mm)
target_link_libraries(raceline PRIVATE raceline_core)

# Tests
add_executable(test_linear tests/test_linear.mm)
target_link_libraries(test_linear PRIVATE raceline_core)

# Hardware probes
add_executable(membw bench/hw/membw.c)
add_executable(nax_probe bench/hw/nax_probe.mm)
target_link_libraries(nax_probe PRIVATE "-framework Metal" "-framework Foundation")
