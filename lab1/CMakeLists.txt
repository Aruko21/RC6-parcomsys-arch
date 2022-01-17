cmake_minimum_required(VERSION 3.15)
project(plateOpenMp)

set(CMAKE_C_STANDARD 99)

set(SRCS src/main.c src/solution.c src/utils.c)
find_package(OpenMP REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(plateOpenMp ${SRCS})
target_link_libraries(plateOpenMp PUBLIC OpenMP::OpenMP_C)