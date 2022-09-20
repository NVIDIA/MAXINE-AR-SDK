set(SOURCE_FILES FaceEngine.cpp
        FaceTrack.cpp
        ../utils/RenderingUtils.cpp
        ../utils/FeatureVertexName.cpp
        ../utils/FeatureVertexName.h
)

set(HEADER_FILES FaceEngine.h)

if(MSVC)
    set(SOURCE_FILES ${SOURCE_FILES} ../../nvar/src/nvARProxy.cpp ../../nvar/src/nvCVImageProxy.cpp)

    # Set Visual Studio source filters
    source_group("Source Files" FILES ${SOURCE_FILES})
    source_group("Header Files" FILES ${HEADER_FILES})
endif(MSVC)


add_executable(FaceTrack ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(FaceTrack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(FaceTrack PUBLIC
    ${SDK_INCLUDES_PATH}
    )

if(MSVC)
    target_link_libraries(FaceTrack PUBLIC
        opencv346
        utils_sample
        )
    set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
    set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
    set_target_properties(FaceTrack PROPERTIES
        FOLDER SampleApps
        VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"  )
elseif(UNIX)
    find_package(PNG  REQUIRED)
    find_package(JPEG REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    target_link_libraries(FaceTrack PUBLIC
        nvARPose
        NVCVImage
        OpenCV
        utils_sample
        )
endif()
