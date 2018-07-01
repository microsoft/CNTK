cmake_minimum_required(VERSION 3.8)

project(CNTK)

include(Tools/cmake/options.cmake REQUIRED)
include(Tools/cmake/common.cmake REQUIRED)
include(Tools/cmake/cntk_common.cmake REQUIRED)

add_subdirectory(bindings/csharp/swig)

include(Tools/cmake/debug.cmake REQUIRED)