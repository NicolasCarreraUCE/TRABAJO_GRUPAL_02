cmake_minimum_required(VERSION 3.25)
project(TRABAJO_GRUPAL_02)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Incluye el módulo CPM
include(cmake/CPM.cmake)

# https://github.com/fmtlib/fmt.git
CPMAddPackage("gh:fmtlib/fmt#10.0.0")

add_executable(TRABAJO_GRUPAL_02 main.cpp)

target_link_libraries(TRABAJO_GRUPAL_02 fmt::fmt)
