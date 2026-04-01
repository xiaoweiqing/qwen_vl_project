cmake_minimum_required(VERSION 3.12)
project(VulkanAiEffects) # Renamed the project for clarity

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan (same as before)
find_package(Vulkan REQUIRED)

# --- NEW: Find the graphics and vision libraries ---
find_package(glfw3 REQUIRED)
#find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)
# --- END NEW ---

# We are now building an executable, not a Python module.
# We will remove the pybind11 parts for now to keep things simple.
add_executable(VulkanAiEffects src/main.cpp) # We will only compile main.cpp for now

# --- UPDATED: Link all the new libraries ---
target_link_libraries(VulkanAiEffects PRIVATE
    Vulkan::Vulkan
    glfw
#   ${OpenCV_LIBS} # <--- THIS IS THE FIX
    glm::glm
)
# --- END UPDATED ---
# --- NEW: Copy shaders to the build directory ---
file(GLOB SHADERS "${CMAKE_SOURCE_DIR}/*.spv")
add_custom_command(
    TARGET VulkanAiEffects POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${SHADERS}
    ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Copying shaders to build directory"
)
