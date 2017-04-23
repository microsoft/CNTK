cmake_minimum_required(VERSION 2.8)

PROJECT(MULTIVERSO)

OPTION(USE_HDFS "won't use hdfs on default, set ON to enable" OFF)
OPTION(TEST "Build all tests." ON)
OPTION(USE_ZMQ "weather to build with ZeroMQ.(default: OFF)" OFF)
OPTION(INSTALL_MULTIVERSO "whether install Multiverso to /usr/local/lib" ON)
option(ENABLE_DCASGD "Build with DC-ASGD supported" ON)

find_package(MPI REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if(USE_HDFS)
    ADD_DEFINITIONS(-DMULTIVERSO_USE_HDFS)
    SET(JVM_LIB /usr/local/java/default/jre/lib/amd64/server)
    SET(HDFS_LIB /usr/local/hadoop/lib/native)
    LINK_DIRECTORIES(${HDFS_LIB})
    LINK_DIRECTORIES(${JVM_LIB})
endif(USE_HDFS)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(MULTIVERSO_DIR ${PROJECT_SOURCE_DIR})
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(Test)
ADD_SUBDIRECTORY(Test/unittests)
ADD_SUBDIRECTORY(Applications/WordEmbedding)
ADD_SUBDIRECTORY(Applications/LogisticRegression)

if(USE_INSTALL_MULTIVERSO)
    install (DIRECTORY ${PROJECT_SOURCE_DIR}/include/multiverso DESTINATION include)
endif(USE_INSTALL_MULTIVERSO)


# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
