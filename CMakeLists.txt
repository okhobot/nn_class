cmake_minimum_required(VERSION 3.22)

set(OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Set main output directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

# Set library output directory
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIR})

# Set archive output directory
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUT_DIR})



set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-s")

add_subdirectory(opencl_wrapper)
add_subdirectory(nn_class)


