#######################
# Interface to OpenCV #
#######################

if(MSVC)
	if(CMAKE_CL_64)
		set(OpenCV_ARCH x64)
	elseif((CMAKE_GENERATOR MATCHES "ARM") OR ("${arch_hint}" STREQUAL "ARM") OR (CMAKE_VS_EFFECTIVE_PLATFORMS MATCHES "ARM|arm"))
		# see Modules/CmakeGenericSystem.cmake
		set(OpenCV_ARCH ARM)
	else()
		set(OpenCV_ARCH x86)
	endif()
	if(MSVC_VERSION GREATER_EQUAL 1920)
		#set(OpenCV_RUNTIME vc16)
		message("No Visual Studio 2019 OpenCV library available; trying 2017 library instead")
		set(OpenCV_RUNTIME vc15)
	elseif(MSVC_VERSION GREATER_EQUAL 1910)
		set(OpenCV_RUNTIME vc15)
	elseif(MSVC_VERSION GREATER_EQUAL 1900)
		set(OpenCV_RUNTIME vc14)
	else()
		message("MSVC_VERSION ${MSVC_VERSION} is not accommodated")
	endif()
endif()

add_library(opencv346 INTERFACE)
set(OpenCV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv/include ${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv2)
target_include_directories(opencv346 INTERFACE ${OpenCV_INCLUDE_DIR})
target_link_libraries(opencv346 INTERFACE optimized ${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib/opencv_world346.lib)

####################
# Interface to GLM #
####################
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    PATH_SUFFIXES include
    PATHS
    glm
    /usr/local/include/glm
)
add_library(GLM INTERFACE)
target_include_directories(GLM INTERFACE ${GLM_INCLUDE_DIR})
#message( "GLM_INCLUDE_DIR          is ${GLM_INCLUDE_DIR}" )

