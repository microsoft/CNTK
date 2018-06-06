# This file contains definitions for C# settings that impact code generation that are 
# generically applicable across a variety of different projects. Project-specific customizations 
# should be defined in a separate file.

include(${CMAKE_CURRENT_LIST_DIR}/common.cmake REQUIRED)

# Ensure that dotnet is available on Linux. Windows builds rely on MSBuild instead.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    EnsureTools(dotnet;)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    EnsureTools(dotnet;)
else()
    message(FATAL_ERROR "The CMAKE_HOST_SYSTEM_NAME value of '${CMAKE_HOST_SYSTEM_NAME}' is not recognized.")
endif()

set(CMAKE_CONFIGURATION_TYPES Debug;Release)

# If CMAKE_BUILD_TYPE is defined, we are looking at a single configuration and can extract information directly from this value.
# If it is not defined, then we need to use a dynamic generator.
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    set(config "${CMAKE_BUILD_TYPE}")
    string(COMPARE EQUAL "${config}" "Debug" is_debug)

elseif(CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    # This code isn't quite right. What we actually want to do is say that CMAKE_BUILD_TYPE must be defined
    # for any generator that doesn't support the use of CONFIG as a generator expression. I haven't found a way
    # to detect that in the cmake logic.
    message(FATAL_ERROR "'CMAKE_BUILD_TYPE' must be defined")
    
else()
    set(config $<CONFIG>)
    set(is_debug $<CONFIG:Debug>)
    
endif()

