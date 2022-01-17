cmake_minimum_required(VERSION 3.19)
project(plateCUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)

set(SRCS src/solution.cu src/utils.cu)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(plateCUDA main.cu ${SRCS})

set_target_properties(
        plateCUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)