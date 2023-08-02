cmake_minimum_required(VERSION 3.25)
project(TRABAJO_GRUPAL_02)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Incluye el módulo CPM
include(cmake/CPM.cmake)

# https://github.com/fmtlib/fmt.git
CPMAddPackage("gh:fmtlib/fmt#10.0.0")

find_package(MPI REQUIRED)
message(STATUS "******Run: ${MPIEXEC}")

# --- INCLUDE libpng LIBRARY TO PROJECT
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(TRABAJO_GRUPAL_02 main.cpp)

target_link_libraries(TRABAJO_GRUPAL_02 fmt::fmt MPI::MPI_CXX ${PNG_LIBRARY})
