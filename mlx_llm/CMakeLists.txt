cmake_minimum_required(VERSION 3.27)
project(MLX_llm LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(MLX CONFIG REQUIRED)

# ----------------------------- Build Directory -----------------------------
set(BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)  # Specify the build directory

add_library(mlx_llm STATIC llm.cpp)
target_link_libraries(mlx_llm PUBLIC mlx)
target_include_directories(mlx_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ----------------------------- Output Directory -----------------------------
set_target_properties(mlx_llm PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${BUILD_DIR})  # Set the output directory for the library