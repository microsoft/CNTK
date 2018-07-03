# This file contains definitions for C++ settings that impact the Application Binary Interface (ABI) that are
# generally applicable across a variety of different projects. Project-specific customizations should be defined
# in a separate file.

option( 
    CODE_COVERAGE
    "Produce builds that can be used to extract code coverage information."
    "OFF"
)

option(
    SUPPORT_AVX2
    "Produce builds that support Advanced Vector Extensions."
    "OFF"
)

set(CMAKE_CONFIGURATION_TYPES Debug;Release;Release_NoOpt)

# If CMAKE_BUILD_TYPE is defined, we are looking at a single configuration and can extract information directly from this value.
# If it is not defined, then we need to use a dynamic generator.
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    set(config "${CMAKE_BUILD_TYPE}")
    string(COMPARE EQUAL "${config}" "Debug" is_debug)
    string(COMPARE EQUAL "${config}" "Release_NoOpt" is_release_noopt)
    
elseif(CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    # This code isn't quite right. What we actually want to do is say that CMAKE_BUILD_TYPE must be defined
    # for any generator that doesn't support the use of CONFIG as a generator expression. I haven't found a way
    # to detect that in the cmake logic.
    message(FATAL_ERROR "'CMAKE_BUILD_TYPE' must be defined")

else()
    set(config $<CONFIG>)
    set(is_debug $<CONFIG:Debug>)
    set(is_release_noopt $<CONFIG:Release_NoOpt>)
    
endif()

if(MSVC)
    # ----------------------------------------------------------------------
    # |
    # | Microsoft Visual Studio
    # |    
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # | Preprocessor Definitions
    add_definitions(
        -DUNICODE;                                                                              # Use UNICODE character set
        -D_UNICODE;                                                                             # Use UNICODE character set
    )                                                       
    
    # Debug-specific flags
    foreach(definition
        -DDEBUG                                                                                 # Debug mode
        -D_DEBUG                                                                                # Debug mode
    )
        # Note that add_definitions isn't able to handle generator expressions, so we
        # have to do the generation manually.
        string(APPEND CMAKE_CXX_FLAGS_DEBUG " ${definition}")
    endforeach()
    
    # ----------------------------------------------------------------------
    # | Compiler Flags
    add_compile_options(                                                                        # Option
                                                                                                # ------------------------------------
        /bigobj                                                                                 # Increase number of sections in object file
        /fp:except-                                                                             # Enable Floating Point Exceptions: No
        /fp:fast                                                                                # Floating Point Model: Fast
        /GR                                                                                     # Run Time Type Information (RTTI)
        /MP                                                                                     # Build with multiple processes
        /openmp                                                                                 # OpenMP 2.0 Support
        /sdl                                                                                    # Enable Additional Security Checks
        /W4                                                                                     # Warning level 4
        /WX                                                                                     # Warning as errors
    )                         
    
                                                                                                # Option                                Debug                   Release                 Release_NoOpt
                                                                                                # ------------------------------------  ----------------------  ----------------------  ----------------------
    add_compile_options($<$<NOT:${is_debug}>:/Gy>)                                              # Enable Function-Level Linking         <Default>               Yes                     Yes
    add_compile_options($<$<NOT:${is_debug}>:/Oi>)                                              # Enable Intrinsic Functions            <Default>               Yes                     Yes
    add_compile_options($<$<NOT:$<OR:${is_debug},${is_release_noopt}>>:/Ot>)                    # Favor Size or Speed                   <Default>               fast                    <Default>
    add_compile_options($<$<NOT:${is_debug}>:/Qpar>)                                            # Enable Parallel Code Generation       <Default>               Yes                     Yes
    add_compile_options($<IF:${is_debug},/ZI,/Zi>)                                              # Program Database                      Edit & Continue         Standard                Standard
    
    # ----------------------------------------------------------------------
    # | Linker Flags
    if(${CODE_COVERAGE})
        foreach(linker_flag                                                                     # Option
                                                                                                # ------------------------------------
            /PROFILE                                                                            # Enable profiling
        )
            string(APPEND CMAKE_EXE_LINKER_FLAGS " ${linker_flag}")
            string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${linker_flag}")
        endforeach()
    endif()

    # Release-specific linker flags
    foreach(linker_flag 
        /DEBUG                                                                                  # Generate Debug Information
        /OPT:ICF                                                                                # Enable COMDAT Folding
        /OPT:REF                                                                                # References
    )
        # Note that there isn't a cmake method called add_linker_options that is able to handle 
        # generator expressions, so we have to do the generation manually.
        string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " ${linker_flag}")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " ${linker_flag}")
    endforeach()

else()
    # ----------------------------------------------------------------------
    # |
    # | GCC(-like)
    # |    
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # | Preprocessor Definitions
    add_definitions(
        -D__USE_XOPEN2K
        -D_POSIX_SOURCE
        -D_XOPEN_SOURCE=600
        -DNO_SYNC
    )

    # Debug-specific flags
    foreach(definition
        -D_DEBUG                                                                                # Debug mode
    )
        # Note that add_definitions isn't able to handle generator expressions, so we
        # have to do the generation manually.
        string(APPEND CMAKE_CXX_FLAGS_DEBUG " ${definition}")
    endforeach()

    # ----------------------------------------------------------------------
    # | Compiler Flags
    add_compile_options(                                                                        # Option
                                                                                                # ------------------------------------
        -fcheck-new                                                                             # Check the return value of new in C++.
        -fopenmp                                                                                # Enable OpenMP
        -fpermissive                                                                            # Downgrade conformance errors to warnings.
        -fPIC                                                                                   # Generate position-independent code if possible (large mode).
        -msse4.1                                                                                # Support MMX, SSE, SSE2, SSE3, SSSE3 and SSE4.1 built-in functions and code generation.
        -std=c++11                                                                              # Conform to the ISO 2011 C standard.
        -Wall                                                                                   # Enable most warning messages.
        -Wextra                                                                                 # Print extra (possibly unwanted) warnings. 
        # TODO: -Werror                                                                                 # Treat all warnings as errors.
    )

    if(${CODE_COVERAGE})
        add_compile_options(
            -fprofile-arcs                                                                      # Insert arc-based program profiling code.
            -ftest-coverage                                                                     # Create data files needed by "gcov".
        )
    endif()

    if(${SUPPORT_AVX2})
        add_compile_options(
            -mavx2                                                                              # Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation.
        )
    endif()
                                                                                                # Option                                Debug                   Release                 Release_NoOpt
                                                                                                # ------------------------------------  ----------------------  ----------------------  ----------------------
    add_compile_options($<$<NOT:$<OR:${is_debug},${is_release_noopt}>>:-O4>)                    # Set optimization level                <Default>               4                       <Default>

    # ----------------------------------------------------------------------
    # | Linker Flags
    foreach(linker_flag                                                                         # Option
                                                                                                # ------------------------------------
        # -rdynamic                                                                             # ???? -dynamicbase?
    )
        # Note that add_definitions isn't able to handle generator expressions, so we
        # have to do the generation manually.
        string(APPEND CMAKE_EXE_LINKER_FLAGS " ${linker_flag}")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${linker_flag}")
    endforeach()

    if(${CODE_COVERAGE})
        foreach(linker_flag                                                                     
            --coverage                                                                          # ????
            -lgcov                                                                              # Link with gcov libraries
            
        )
            # Note that add_definitions isn't able to handle generator expressions, so we
            # have to do the generation manually.
            string(APPEND CMAKE_EXE_LINKER_FLAGS " ${linker_flag}")
            string(APPEND CMAKE_SHARED_LINKER_FLAGS " ${linker_flag}")
        endforeach()
    endif()

endif()

# Define the Release_NoOpt configuration in terms of Release. Differentiation between the configurations 
# are handled in add_compile_options above.
set(CMAKE_CXX_FLAGS_RELEASE_NOOPT ${CMAKE_CXX_FLAGS_RELEASE})
set(CMAKE_EXE_LINKER_FLAGS_RELEASE_NOOPT ${CMAKE_EXE_LINKER_FLAGS_RELEASE})
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE_NOOPT ${CMAKE_SHARED_LINKER_FLAGS_RELEASE})
