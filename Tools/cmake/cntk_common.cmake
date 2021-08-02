set(CNTK_VERSION 2.7.0)

# ----------------------------------------------------------------------
# UnsupportedComponentError
#   Issues a fatal error when attempting to generate projects for use in unverified environments.
#
function(UnsupportedComponentError value component)
    message(FATAL_ERROR 
    
"CNTK has not been verified with the version '${value}' of the ${component}.\n\
\
However, this does not mean that CNTK doesn't work with this component. If you are feeling adventureous, you can add '${value}' to the list of supported ${component} items, build and test CNTK, and report your findings to https://github.com/Microsoft/CNTK/issues (we welcome contributions from the community!). If you believe that this component should be supported, please create an issue at https://github.com/Microsoft/CNTK/issues.
"
            )
endfunction()
          
# ----------------------------------------------------------------------
if(MSVC)
    set(CNTK_SUPPORTED_PLATFORMS
            x64;
    )
    
    if(${CMAKE_VS_PLATFORM_NAME} IN_LIST CNTK_SUPPORTED_PLATFORMS)
        # There doesn't seem to be a NOT operator, thus the wonky syntax
    else()
        UnsupportedComponentError(${CMAKE_VS_PLATFORM_NAME} "development platform")
    endif()
    
    set(CNTK_SUPPORTED_MSVC_VERSIONS
            1911;                           # Visual Studio 2017 version 15.3
    )
    
    if(${MSVC_VERSION} IN_LIST CNTK_SUPPORTED_MSVC_VERSIONS)
        # There doesn't seem to be a NOT operator, thus the wonky syntax
    else()
        UnsupportedComponentError(${MSVC_VERSION} "Visual Studio Compiler")
    endif()
    
    set(CNTK_SUPPORTED_WINDOWS_SKDS
            10.0.17134.0;                   # Windows 10 SDK for April 2018 Update, version 1803
    )
    
    if(${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION} IN_LIST CNTK_SUPPORTED_WINDOWS_SKDS)
        # There doesn't seem to be a NOT operator, thus the wonky syntax
    else()
        UnsupportedComponentError(${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION} "Windows SDK")
    endif()
endif()
