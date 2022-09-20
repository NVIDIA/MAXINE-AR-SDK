set(SOURCE_FILES GazeEngine.cpp 
    GazeRedirect.cpp 
    ../utils/RenderingUtils.cpp 
    ../utils/FeatureVertexName.cpp
    ../utils/FeatureVertexName.h
)
set(HEADER_FILES GazeEngine.h )
if(MSVC)
    set(SOURCE_FILES ${SOURCE_FILES} ../../nvar/src/nvARProxy.cpp  ../../nvar/src/nvCVImageProxy.cpp)
    set(HEADER_FILES ${HEADER_FILES} ../utils/RenderingUtils.h)
endif(MSVC)

# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

add_executable(GazeRedirect ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(GazeRedirect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(GazeRedirect PUBLIC
 ${SDK_INCLUDES_PATH}
)

if(MSVC)
    set(ARSDK_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)
    set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
    set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
    set(CMD_ARG_STR "--model_path=\"${CMAKE_CURRENT_SOURCE_DIR}/../../bin/models\"")
    target_link_libraries (GazeRedirect PUBLIC
        ${OPENGL_gl_LIBRARY}
        opencv346
        utils_sample
    )

    set_target_properties(GazeRedirect PROPERTIES 
        FOLDER SampleApps 
        VS_DEBUGGER_ENVIRONMENT "${PATH_STR}" 
        VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
    )
elseif(UNIX)
    find_package(PNG  REQUIRED)
    find_package(JPEG REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    target_link_libraries(GazeRedirect PUBLIC
        nvARPose
        NVCVImage
        OpenCV
        utils_sample
    )
endif(MSVC)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")