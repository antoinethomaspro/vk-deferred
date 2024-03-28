cmake_minimum_required(VERSION 3.8)
project(deferred_renderer)

# Set C++ standard (optional)
set(CMAKE_CXX_STANDARD 20)

# Set the external directory
set(EXTERNAL "third_party")

#for glm
include_directories("${EXTERNAL}/include")


# Set GLFW root directory
set(GLFW_ROOT_DIR "${EXTERNAL}/glfw")
# Set GLFW options to control the build configuration
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)

# Add GLFW as a subdirectory and build it
add_subdirectory(${GLFW_ROOT_DIR})

# Include GLFW headers directory
include_directories("${GLFW_ROOT_DIR}/include")

# Add vk-bootstrap subdirectory
add_subdirectory(third_party/vk-bootstrap)
include_directories("${EXTERNAL}/third_party/vk-bootstrap/src")

add_subdirectory(third_party/fmt)

# Add all files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Add an executable target
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# Find our Vulkan!
find_package(Vulkan REQUIRED)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${Vulkan_LIBRARIES})

#target_link_libraries(${CMAKE_PROJECT_NAME} glfw ${GLFW_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME} glfw)

# Link against vk-bootstrap
target_link_libraries(${CMAKE_PROJECT_NAME} vk-bootstrap::vk-bootstrap)
target_link_libraries(${CMAKE_PROJECT_NAME} fmt::fmt)