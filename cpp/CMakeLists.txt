cmake_minimum_required(VERSION 3.16)
project(vpxc LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VPXC_NATIVE_ARCH "Enable host CPU SIMD/BMI instructions (-march=native)" ON)
if(VPXC_NATIVE_ARCH)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
endif()

add_library(vpxc STATIC
  src/bitio.cpp
  src/entropy.cpp
  src/wsil.cpp
  src/color.cpp
  src/lattice.cpp
  src/issm.cpp
  src/spatial.cpp
  src/coarse.cpp
  src/matching.cpp
  src/acquisition.cpp
  src/trees.cpp
  src/bitsback.cpp
  src/mdl.cpp
  src/slgf.cpp
  src/slgf_v3.cpp
  src/wp.cpp
  src/frontier.cpp
  src/codec.cpp
)
find_package(Threads REQUIRED)
target_include_directories(vpxc PUBLIC include)
target_link_libraries(vpxc PUBLIC z Threads::Threads)
target_compile_options(vpxc PRIVATE -Wall -Wextra)
if(VPXC_NATIVE_ARCH AND COMPILER_SUPPORTS_MARCH_NATIVE)
  target_compile_options(vpxc PRIVATE -march=native)
endif()
set_target_properties(vpxc PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(vpxc_native SHARED src/c_api.cpp)
target_link_libraries(vpxc_native PRIVATE vpxc)
set_target_properties(vpxc_native PROPERTIES OUTPUT_NAME vpxc_native)
