cmake_minimum_required(VERSION 3.27)
project(MLX_llm LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)


add_library(mlx_llm STATIC llm.h)
target_include_directories (mlx_llm PUBLIC${CMAKE_CURRENT_SOURCE_DIR})