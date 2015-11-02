# vim:set expandtab shiftwidth=4 tabstop=4:

cmake_minimum_required(VERSION 2.8)

project(CNTKCMake CXX)

find_package(MPI)

if(WIN32)

add_definitions(-DUNICODE -D_UNICODE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

set(CMAKE_CXX_STANDARD 11)

set(ACML_PATH C:/AMD/acml5.3.1/ifort64_mp)

else()

set(ACML_PATH /usr/local/acml5.3.1/ifort64)

# Note: I'm on an older version on Linux, no CMAKE_CXX_STANDARD ?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -std=c++0x -std=c++11 -fopenmp -fpermissive -fPIC -Werror -fcheck-new")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3 -std=c++11 -fpermissive")
# TODO less flags?

endif()

add_definitions(-DCPUONLY)

set(CMAKE_BUILD_TYPE "Release" "Build Type")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(
    Math/Math
    Common/Include
    ${ACML_PATH}/include
    ${MPI_CXX_INCLUDE_PATH}
    MachineLearning/CNTK
    MachineLearning/CNTKComputationNetworkLib
    MachineLearning/CNTKSGDLib
    MachineLearning/CNTKSequenceTrainingLib
    BrainScript)

add_library(Common STATIC
    Common/BestGpu.cpp
    Common/ConfigFile.cpp
    Common/DataReader.cpp
    Common/DataWriter.cpp
    Common/DebugUtil.cpp
    Common/Eval.cpp
    Common/File.cpp
    Common/TimerUtility.cpp
    Common/fileutil.cpp)

if(NOT WIN32)
target_link_libraries(Common
    dl)
endif()

add_library(Math SHARED
    Math/Math/CPUMatrix.cpp
    Math/Math/CPUSparseMatrix.cpp
    Math/Math/MatrixQuantizer.cpp
    Math/Math/MatrixQuantizerCPU.cpp
    Math/Math/QuantizedMatrix.cpp
    Math/Math/Matrix.cpp
    Math/Math/CUDAPageLockedMemAllocator.cpp
    Math/Math/NoGPU.cpp)
set_target_properties(Math PROPERTIES
    COMPILE_DEFINITIONS MATH_EXPORTS)

add_library(acml SHARED IMPORTED)

if(WIN32)

set_target_properties(acml PROPERTIES
    IMPORTED_LOCATION ${ACML_PATH}/lib/libacml_mp_dll.dll
    IMPORTED_IMPLIB ${ACML_PATH}/lib/libacml_mp_dll.lib)

else()

set_target_properties(acml PROPERTIES
    IMPORTED_LOCATION ${ACML_PATH}/lib/libacml.so)

endif()

target_link_libraries(Math
    acml
    Common)

add_library(UCIFastReader SHARED
    DataReader/UCIFastReader/Exports.cpp
    DataReader/UCIFastReader/UCIFastReader.cpp
    DataReader/UCIFastReader/UCIParser.cpp)

target_link_libraries(UCIFastReader
    Common
    Math)

# TODO default on Win anyway:
set_target_properties(UCIFastReader PROPERTIES
    PREFIX "")

add_executable(cntk
    MachineLearning/CNTK/CNTK.cpp
    MachineLearning/CNTK/ModelEditLanguage.cpp
    MachineLearning/CNTK/NetworkDescriptionLanguage.cpp
    MachineLearning/CNTK/SimpleNetworkBuilder.cpp
    MachineLearning/CNTK/SynchronousExecutionEngine.cpp
    MachineLearning/CNTK/tests.cpp
    MachineLearning/CNTKComputationNetworkLib/ComputationNode.cpp
    MachineLearning/CNTKComputationNetworkLib/ComputationNetwork.cpp
    MachineLearning/CNTKComputationNetworkLib/ComputationNetworkEvaluation.cpp
    MachineLearning/CNTKComputationNetworkLib/ComputationNetworkAnalysis.cpp
    MachineLearning/CNTKComputationNetworkLib/ComputationNetworkEditing.cpp
    MachineLearning/CNTKComputationNetworkLib/ComputationNetworkBuilder.cpp
    MachineLearning/CNTKComputationNetworkLib/NetworkBuilderFromConfig.cpp
    MachineLearning/CNTKSGDLib/Profiler.cpp
    MachineLearning/CNTKSGDLib/SGD.cpp
    MachineLearning/CNTKSequenceTrainingLib/latticeforwardbackward.cpp
    MachineLearning/CNTKSequenceTrainingLib/parallelforwardbackward.cpp
    BrainScript/BrainScriptEvaluator.cpp
    BrainScript/BrainScriptParser.cpp
    BrainScript/BrainScriptTest.cpp
    MachineLearning/CNTK/ExperimentalNetworkBuilder.cpp
    MachineLearning/CNTKSequenceTrainingLib/latticeNoGPU.cpp
    Common/MPIWrapper.cpp)

link_directories(${ACML_PATH}/lib)

if(WIN32)

target_link_libraries(cntk
    Math
    acml
    ${MPI_CXX_LIBRARIES})

else()

# TODO m and pthread required or implicit already?
target_link_libraries(cntk
    Math
    acml
    m
    pthread
    ${MPI_CXX_LIBRARIES})

endif()

if(WIN32)

foreach(dll libacml_mp_dll.dll libifcoremd.dll libiomp5md.dll svml_dispmd.dll libmmd.dll)
    add_custom_command(TARGET cntk POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ACML_PATH}/lib/${dll}
            $<TARGET_FILE_DIR:cntk>)
endforeach(dll)

add_custom_command(TARGET cntk POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${MPI_CXX_LIBRARIES}
        $<TARGET_FILE_DIR:cntk>)

endif()

enable_testing()

# Right now, we use Boost only for testing
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED
  COMPONENTS
  unit_test_framework REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(MathTests
    Tests/UnitTests/MathTests/CPUMatrixTests.cpp)

target_link_libraries(MathTests
    Math
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    acml)

add_test(NAME MathTests COMMAND MathTests)

# Random notes follow

# cmake.org/Wiki/CMake_RPATH_handling
# TODO needs change
#set(CMAKE_SKIP_BUILD_RPATH FALSE)
#set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#set(CMAKE_INSTALL_RPATH "")
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# TODO per-target include paths?
# TODO cleanup
# Defined by default:
#set_target_properties(UCIFastReader PROPERTIES
#    COMPILE_DEFINITIONS DATAREADER_EXPORTS)

#message("${MPI_CXX_LINK_FLAGS}")
#message("${MPI_CXX_LIBRARIES}")

#    MPI_<lang>_LINK_FLAGS      Linking flags for MPI programs

#set(BLA_VENDOR ACML_MP)
#find_package(BLAS)
#set(_ACML_ROOT /usr/local/acml5.3.1)
#set(BLAS_LIBRARIES /usr/local/acml5.3.1)
#set(BLAS_ACML_LIB_DIRS /usr/local/acml5.3.1)
#set(BLAS_ACML_MP_LIB_DIRS /usr/local/acml5.3.1)
#set(BLAS_ACML_LIB_DIRS

# CMAKE 3 has more smarts around this
#set_property(TARGET Common PROPERTY CXX_STANDARD_REQUIRED ON)

#add_library(ACML SHARED IMPORTED0
#set_property(TARGET ssl PROPERTY IMPORTED_LOCATION /usr/lib/...)

# TODO
# set(CMAKE_CXX_FLAGS "  ")
# set(BLAS_PATH "/usr/local/acml5.3.1/ifort64")
#if( WIN32 )
#else()

# TODO check out BLAS detection (i) with new version (ii) on Windows.
#Math/Math/Math.vcxproj:      <Command>xcopy /D /I /Y "$(ACML_PATH)\lib\*.dll" $(OutputPath)</Command>

#MESSAGE("$ENV{ACML_PATH}")
#set(ACML_PATH $ENV{ACML_PATH})

