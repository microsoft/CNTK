# ----------------------------------------------------------------------
# EnsureProperties
#   Validates that all properties have been defined.
#
function(EnsureProperties)
    foreach(property ${ARGN})
        if(NOT DEFINED "${property}")
            message(FATAL_ERROR 
        
"'${property}' is not defined.\n\
\
This property must be defined to correctly generate content. There is a configuration problem if you see this error when generating the core CNTK project. If you are generating this file in isolation, you can add the entry within the cmake GUI or specify it on the cmake command line with -Dvar=value.
"               
                   )
        endif()
    endforeach()
endfunction()

# ----------------------------------------------------------------------
# EnsureTools
#   Validates that all tools are available. This is a macro to ensure that the property '<tool_name>_binary' is available within
#   the namespace of calling scripts.
#
macro(EnsureTools)
    foreach(tool ${ARGN})
        find_program(${tool}_binary ${tool})
        if(${tool}_binary STREQUAL ${tool}_binary-NOTFOUND)
            message(FATAL_ERROR
    
"The tool '${tool}' was not found.\n\
\
Make sure that this tool is available in a directory available via the 'PATH' environment variable and run cmake again.
"
               )
        endif()
    endforeach()
endmacro()
