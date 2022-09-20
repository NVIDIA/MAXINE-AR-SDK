cmake_minimum_required(VERSION 3.10.0)

# Set path where samples will be installed
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR} CACHE PATH "Path to where the samples will be installed")

project(NvAR_SDK CXX)

set(CMAKE_CONFIGURATION_TYPES "Release")

# Require C++11 and disable non-standard extensions

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if(MSVC)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(SDK_INCLUDES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/nvar/include)

    # Add target for nvARPose
    add_library(nvARPose INTERFACE)
    target_include_directories(nvARPose INTERFACE ${SDK_INCLUDES_PATH})

    # Add target for NVCVImage
    add_library(NVCVImage INTERFACE)
    target_include_directories(NVCVImage INTERFACE ${SDK_INCLUDES_PATH})

else()
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    # Add target for ARSDK
    add_library(nvARPose INTERFACE)

    # found in different locations depending on type of package
    find_path(ARSDK_INCLUDES
        NAMES nvAR.h
        PATHS
        /usr/local/ARSDK/include
        /usr/include/x86_64-linux-gnu
        /usr/include
        ${CMAKE_CURRENT_SOURCE_DIR}/nvar/include
        REQUIRED
        )

    target_include_directories(nvARPose INTERFACE ${ARSDK_INCLUDES})

    find_library(ARSDK_LIB
        NAMES libnvARPose.so
        PATHS
        /usr/local/ARSDK/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/bin
        REQUIRED
        NO_DEFAULT_PATH)

    target_link_libraries(nvARPose INTERFACE "${ARSDK_LIB}")

    message(STATUS "ARSDK_LIB: ${ARSDK_LIB}")


    # Add target for NVCVImage
    add_library(NVCVImage INTERFACE)

    # found in different locations depending on type of package
    find_path(NVCVImage_INCLUDES
        NAMES nvCVImage.h
        PATHS
        /usr/local/ARSDK/include
        /usr/include/x86_64-linux-gnu
        /usr/include
        ${CMAKE_CURRENT_SOURCE_DIR}/nvar/include
        REQUIRED
        )

    target_include_directories(NVCVImage INTERFACE ${NVCVImage_INCLUDES})


    find_library(NVCVImage_LIB
        NAMES libNVCVImage.so
        PATHS
        /usr/local/ARSDK/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/bin
        REQUIRED
        NO_DEFAULT_PATH)

    target_link_libraries(NVCVImage INTERFACE "${NVCVImage_LIB}")

    message(STATUS "NVCVImage_LIB: ${NVCVImage_LIB}")
    message(STATUS "NVCVImage_INCLUDES_PATH: ${NVCVImage_INCLUDES}")


endif()

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)

# Set common build path for all targets
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


set(ENABLE_SAMPLES TRUE)
add_subdirectory(samples)
