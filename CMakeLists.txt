cmake_minimum_required(VERSION 3.15)
project(dcsam)

# Set some compilation options.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  # Options: Debug, Release, MinSizeRel, RelWithDebInfo
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Add option to enable testing
option(DCSAM_ENABLE_TESTS "Enable tests" ON)

# External package dependencies.
find_package(GTSAM REQUIRED)
# We use Eigen bundled with GTSAM, so we don't need to find it separately.

# add_definitions(-march=native)
# add_definitions(-std=c++1z)

add_library(dcsam SHARED)
target_sources(dcsam PRIVATE src/DCSAM.cpp src/HybridFactorGraph.cpp)
target_include_directories(dcsam PUBLIC include)
target_link_libraries(dcsam PUBLIC gtsam)
target_compile_options(dcsam PRIVATE -Wall -Wpedantic -Wextra)

# Make library accessible to other cmake projects
export(PACKAGE dcsam)
export(TARGETS dcsam FILE dcsamConfig.cmake)

# Include unit tests directory to the project.
if(DCSAM_ENABLE_TESTS)
  message(STATUS "Testing enabled. Building tests.")
  enable_testing()
  add_subdirectory(tests)
endif()
