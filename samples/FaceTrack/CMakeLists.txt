set(SOURCE_FILES FaceEngine.cpp 
FaceTrack.cpp 
../utils/RenderingUtils.cpp 
../../nvar/src/nvARProxy.cpp
../utils/FeatureVertexName.cpp
../utils/FeatureVertexName.h
)
set(HEADER_FILES FaceEngine.h)

# Set Visual Studio source filters
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

add_executable(FaceTrack ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(FaceTrack PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(FaceTrack PUBLIC
 ${SDK_INCLUDES_PATH}
)
target_link_libraries(FaceTrack PUBLIC
	opencv346
	utils_sample
	GLM
)

set(ARSDK_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)
set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
set(PATH_STR "PATH=%PATH%" ${ARSDK_PATH_STR} ${OPENCV_PATH_STR})
set(CMD_ARG_STR "--model_path=\"${CMAKE_CURRENT_SOURCE_DIR}/../../bin/models\"")
set_target_properties(FaceTrack PROPERTIES 
	FOLDER SampleApps 
	VS_DEBUGGER_ENVIRONMENT "${PATH_STR}" 
	VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"
)


