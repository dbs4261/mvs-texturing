cmake_minimum_required(VERSION 3.14)
project(Texturing LANGUAGES CXX)
set(ignoreMe "${CMAKE_C_COMPILER}")

set(TEXRECON_VERSION 21.6.3)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui, ccmake
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED yes)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-msse HAS_SSE)
if(NOT HAS_SSE)
    message(FATAL_ERROR "Compiler does not support sse intrinsics!")
endif()
check_cxx_compiler_flag(-mfpmath=sse HAS_SSE_FP)
if(NOT HAS_SSE_FP)
    message(FATAL_ERROR "Compiler does not support floating point sse intrinsics!")
endif()
check_cxx_compiler_flag(-funroll-loops HAS_UNROLLING)

add_library(tex_settings INTERFACE)
target_compile_options(tex_settings INTERFACE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wundef -pedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Weverything -pedantic>
    $<$<BOOL:${HAS_SSE}>:-msse>
    $<$<BOOL:${HAS_SSE_FP}>:-mfpmath=sse>
    $<$<BOOL:${HAS_UNROLLING}>:-funroll-loops>
)

find_package(OpenMP)

add_definitions(-DEIGEN_MPL2_ONLY)
find_package(Eigen3 CONFIG REQUIRED)

find_package(mve REQUIRED)
find_package(mapmap REQUIRED)
find_package(rayint REQUIRED)

add_subdirectory(libs)
add_subdirectory(apps)
