######################
# The Expression app #
######################
set(APP_SRCS
    ExpressionApp.cpp
    MeshRenderer.cpp MeshRenderer.h
    DirectoryIterator.cpp DirectoryIterator.h
    BackEndOpenGL/GLMaterial.cpp BackEndOpenGL/GLMaterial.h
    BackEndOpenGL/GLMesh.cpp BackEndOpenGL/GLMesh.h
    BackEndOpenGL/GLShaders.cpp BackEndOpenGL/GLShaders.h
    BackEndOpenGL/GLSpectrum.h
    BackEndOpenGL/SimpleFaceModel.h
    BackEndOpenGL/OpenGLMeshRenderer.cpp BackEndOpenGL/OpenGLMeshRenderer.h
    BackEndOpenGL/FaceIO.cpp BackEndOpenGL/FaceIO.h
)
if(WIN32)
    set(APP_SRCS ${APP_SRCS} nvARProxy.cpp nvCVImageProxy.cpp)
    find_package(OpenGL REQUIRED)
endif(WIN32)

option(ENABLE_UI "Enable UI to adjust rigging" OFF)     # ON still needs some more link debugging

if (${ENABLE_UI})
    set(APP_SRCS ${APP_SRCS}
        ExpressionAppUI.h
        ExpressionAppUI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imgui.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imconfig.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imgui_internal.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imstb_textedit.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imstb_rectpack.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imstb_truetype.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/backends/imgui_impl_glfw.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/backends/imgui_impl_opengl3.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/backends/imgui_impl_opengl3_loader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/misc/cpp/imgui_stdlib.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imgui_tables.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imgui_widgets.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imgui.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/imgui_draw.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/backends/imgui_impl_glfw.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/backends/imgui_impl_opengl3.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking/misc/cpp/imgui_stdlib.cpp
    )
endif (${ENABLE_UI})

add_executable(ExpressionApp ${APP_SRCS})

target_include_directories(ExpressionApp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/BackEndOpenGL
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/glm/include
    ${SDK_INCLUDES_PATH}
)
if (${ENABLE_UI})
    add_definitions("-D_ENABLE_UI")
    target_include_directories(ExpressionApp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/Imgui/imgui-docking
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/nlohmann/json/single_include/nlohmann
    )
endif (${ENABLE_UI})

if(WIN32)
    target_link_libraries(ExpressionApp PUBLIC
        opencv346
        glfw3
        GLAD
        ${OPENGL_gl_LIBRARY}
    )
    target_link_directories(ExpressionApp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/GLAD/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/GLFW/lib
    )
    target_include_directories(ExpressionApp PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/GLAD/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/GLFW/include
    )
    set(OPENCV_PATH_STR ${CMAKE_CURRENT_SOURCE_DIR}/../external/opencv/bin)
    set(PATH_STR "PATH=%PATH%" ${OPENCV_PATH_STR})
    set_target_properties(ExpressionApp PROPERTIES
        FOLDER SampleApps
        VS_DEBUGGER_ENVIRONMENT "${PATH_STR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "${CMD_ARG_STR}"  )
elseif(UNIX)
    #find_package(PNG  REQUIRED)
    #find_package(JPEG REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fpermissive")
    target_link_libraries(ExpressionApp PUBLIC
        nvARPose
        NVCVImage
        OpenCV
        glfw
        OpenGL
        dl
    )
endif()
