cmake_minimum_required(VERSION 4.0)

project(rcgp-samples CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LINKER_TYPE MOLD CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(RCGP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/rcgp" CACHE PATH "Path to the rcgp repository")

find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

add_subdirectory(${RCGP_DIR} rcgp-build)

set(RCGP_SAMPLE_INCLUDES
	${RCGP_DIR}/include
	${RCGP_DIR}/source
)

# Shared sample utilities (glTF loading, etc.)
add_library(sample_common STATIC samples/common/gltf.cpp)
target_include_directories(sample_common PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/samples/common
	${CMAKE_CURRENT_SOURCE_DIR}/ext
)
target_link_libraries(sample_common PUBLIC glm::glm)

# Dear ImGui (core + GLFW/Vulkan backends, dynamic rendering)
add_library(imgui STATIC
	ext/imgui/imgui.cpp
	ext/imgui/imgui_draw.cpp
	ext/imgui/imgui_tables.cpp
	ext/imgui/imgui_widgets.cpp
	ext/imgui/backends/imgui_impl_glfw.cpp
	ext/imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui
	${CMAKE_CURRENT_SOURCE_DIR}/ext/imgui/backends
)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

file(GLOB RCGP_SAMPLE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/samples/*.cpp")

foreach(sample_source ${RCGP_SAMPLE_SOURCES})
	get_filename_component(sample_name ${sample_source} NAME_WE)
	add_executable(${sample_name} ${sample_source})
	target_compile_definitions(${sample_name} PRIVATE RCGP_SUPPORT_GLM)
	target_include_directories(${sample_name} PRIVATE ${RCGP_SAMPLE_INCLUDES})
	target_link_libraries(${sample_name} PRIVATE rcgp glm::glm sample_common imgui)
endforeach()
