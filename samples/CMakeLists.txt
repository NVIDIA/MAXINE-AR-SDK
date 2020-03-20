# Sample apps

add_library(utils_sample INTERFACE)
target_include_directories(utils_sample INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_link_libraries(utils_sample INTERFACE GLM)
add_subdirectory(external)
add_subdirectory(FaceTrack)
