cmake_minimum_required(VERSION 3.16)
project(vpxc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
target_include_directories(vpxc PUBLIC include)
target_link_libraries(vpxc PUBLIC z)
target_compile_options(vpxc PRIVATE -Wall -Wextra)
set_target_properties(vpxc PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(vpxc_native SHARED src/c_api.cpp)
target_link_libraries(vpxc_native PRIVATE vpxc)
set_target_properties(vpxc_native PROPERTIES OUTPUT_NAME vpxc_native)
