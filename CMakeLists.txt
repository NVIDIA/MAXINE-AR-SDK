cmake_minimum_required(VERSION 3.12.0)

# Set path where samples will be installed
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "Path to where the samples will be installed")

project(NvAR_SDK CXX)

set(CMAKE_CONFIGURATION_TYPES "Release")

# Require C++11 and disable non-standard extensions
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)

# Set common build path for all targets
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(SDK_INCLUDES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/nvar/include)

# Add target for nvARPose
add_library(nvARPose INTERFACE)
target_include_directories(nvARPose INTERFACE ${SDK_INCLUDES_PATH})

set(ENABLE_SAMPLES TRUE)
add_subdirectory(samples)

