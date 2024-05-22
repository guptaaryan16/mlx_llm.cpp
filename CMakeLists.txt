cmake_minimum_required(VERSION 3.27)
project(MLX_llm LANGUAGES CXX)

# ----------------------------- Setup -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# ----------------------------- Dependencies -----------------------------
find_package(MLX CONFIG REQUIRED)

# Add an executable target
add_executable(MLX_llm test_nn.cpp)

# Add include headers
target_include_directories(
  MLX_llm PUBLIC ${CMAKE_CURRENT_LIST_DIR}
)


add_subdirectory(
    mlx_llm
)
set_target_properties(mlx_llm PROPERTIES LINKER_LANGUAGE CXX)


# Link to mlx
target_link_libraries(MLX_llm PUBLIC mlx)
set_target_properties(MLX_llm PROPERTIES OUTPUT_NAME MLX_llm_output)