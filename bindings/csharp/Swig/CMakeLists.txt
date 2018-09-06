cmake_minimum_required (VERSION 3.8)

include(../../../Tools/cmake/common.cmake REQUIRED)

EnsureProperties(CNTK_VERSION)
EnsureTools(swig;)

include(../../../Tools/cmake/cpp_common.cmake REQUIRED)

set(binary_name_suffix -${CNTK_VERSION}$<${is_debug}:d>) 
set(binary_name Cntk.Core.CSBinding${binary_name_suffix}) 

# ----------------------------------------------------------------------
# |  SWIG
set(swig_configuration $<IF:${is_debug},Debug,Release>) 
set(swig_output_dir "${CMAKE_CURRENT_SOURCE_DIR}/Generated/${CMAKE_SYSTEM_NAME}/${swig_configuration}") 

set(native_swig_generated_files
        ${CMAKE_CURRENT_SOURCE_DIR}/Generated/cntk_cs_wrap.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Generated/cntk_cs_wrap.cxx
)

set(csharp_swig_generated_files
        ${swig_output_dir}/AdditionalLearningOptions.cs;
        ${swig_output_dir}/Axis.cs;
        ${swig_output_dir}/AxisVector.cs;
        ${swig_output_dir}/BoolVector.cs;
        ${swig_output_dir}/CharVector.cs;
        ${swig_output_dir}/CNTKDictionary.cs;
        ${swig_output_dir}/CNTKLib.cs;
        ${swig_output_dir}/CNTKLibPINVOKE.cs;
        ${swig_output_dir}/Constant.cs;
        ${swig_output_dir}/ConstantVector.cs;
        ${swig_output_dir}/DataType.cs;
        ${swig_output_dir}/DataUnit.cs;
        ${swig_output_dir}/DeviceDescriptor.cs;
        ${swig_output_dir}/DeviceDescriptorVector.cs;
        ${swig_output_dir}/DeviceKind.cs;
        ${swig_output_dir}/DictionaryValue.cs;
        ${swig_output_dir}/DictionaryVector.cs;
        ${swig_output_dir}/DoubleVector.cs;
        ${swig_output_dir}/DoubleVectorVector.cs;
        ${swig_output_dir}/Evaluator.cs;
        ${swig_output_dir}/float16.cs;
        ${swig_output_dir}/FloatVector.cs;
        ${swig_output_dir}/FloatVectorVector.cs;
        ${swig_output_dir}/Function.cs;
        ${swig_output_dir}/FunctionPtrVector.cs;
        ${swig_output_dir}/HTKFeatureConfiguration.cs;
        ${swig_output_dir}/HTKFeatureConfigurationVector.cs;
        ${swig_output_dir}/IntVector.cs;
        ${swig_output_dir}/Learner.cs;
        ${swig_output_dir}/LearnerVector.cs;
        ${swig_output_dir}/MaskKind.cs;
        ${swig_output_dir}/MinibatchData.cs;
        ${swig_output_dir}/MinibatchInfo.cs;
        ${swig_output_dir}/MinibatchSource.cs;
        ${swig_output_dir}/MinibatchSourceConfig.cs;
        ${swig_output_dir}/ModelFormat.cs;
        ${swig_output_dir}/NDArrayView.cs;
        ${swig_output_dir}/NDArrayViewPtrVector.cs;
        ${swig_output_dir}/NDMask.cs;
        ${swig_output_dir}/NDShape.cs;
        ${swig_output_dir}/PaddingMode.cs;
        ${swig_output_dir}/PairDoubleDouble.cs;
        ${swig_output_dir}/PairFloatFloat.cs;
        ${swig_output_dir}/PairIntInt.cs;
        ${swig_output_dir}/PairNDArrayViewPtrNDArrayViewPtr.cs;
        ${swig_output_dir}/PairSizeTDouble.cs;
        ${swig_output_dir}/PairSizeTInt.cs;
        ${swig_output_dir}/PairSizeTSizeT.cs;
        ${swig_output_dir}/Parameter.cs;
        ${swig_output_dir}/ParameterCloningMethod.cs;
        ${swig_output_dir}/ParameterVector.cs;
        ${swig_output_dir}/PoolingType.cs;
        ${swig_output_dir}/ProgressWriter.cs;
        ${swig_output_dir}/ProgressWriterVector.cs;
        ${swig_output_dir}/SizeTVector.cs;
        ${swig_output_dir}/SizeTVectorVector.cs;
        ${swig_output_dir}/StorageFormat.cs;
        ${swig_output_dir}/StreamConfiguration.cs;
        ${swig_output_dir}/StreamConfigurationVector.cs;
        ${swig_output_dir}/StreamInformation.cs;
        ${swig_output_dir}/StringVector.cs;
        ${swig_output_dir}/SWIGTYPE_p_int8_t.cs;
        ${swig_output_dir}/SWIGTYPE_p_void.cs;
        ${swig_output_dir}/TraceLevel.cs;
        ${swig_output_dir}/Trainer.cs;
        ${swig_output_dir}/TrainingParameterScheduleDouble.cs;
        ${swig_output_dir}/UnorderedMapParameterNDArrayViewPtr.cs;
        ${swig_output_dir}/UnorderedMapStreamInformationMinibatchData.cs;
        ${swig_output_dir}/UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr.cs;
        ${swig_output_dir}/UnorderedMapStringDictionaryValue.cs;
        ${swig_output_dir}/UnorderedMapVariableMinibatchData.cs;
        ${swig_output_dir}/UnorderedMapVariableValuePtr.cs;
        ${swig_output_dir}/UnorderedMapVariableVariable.cs;
        ${swig_output_dir}/UnsignedCharVector.cs;
        ${swig_output_dir}/Value.cs;
        ${swig_output_dir}/Variable.cs;
        ${swig_output_dir}/VariableKind.cs;
        ${swig_output_dir}/VariablePair.cs;
        ${swig_output_dir}/VariablePairVector.cs;
        ${swig_output_dir}/VariableVector.cs;
        ${swig_output_dir}/VectorPairSizeTDouble.cs;
)

set(swig_generated_files
        ${native_swig_generated_files}
        ${csharp_swig_generated_files}
)

if(MSVC)
    set(swig_params 
            -D_MSC_VER 
            -Werror
    )
else()
    set(swig_params "")
endif()

# Create the swig command line; it needs to be constructed manually as cmake won't convert path separators for the command if it
# is provided as a single value.
set(swig_command_line 
        ${swig_params}
        -c++ 
        -csharp 
        -namespace CNTK 
        -dllimport 
        \"${binary_name}${CMAKE_SHARED_MODULE_SUFFIX}\"
)

# There has got to be a better way to do this. When ${swig_output_dir} is converted to a native path, the
# embedded generator expression is de-generator-expression-ized and won't expand when used in the custom 
# command below. So, remove the generator expression from the var, convert to the native path, and then
# reapply the generator expression. This only works because we know that the last path component is the 
# generator expression.
get_filename_component(cmake_swig_output_dir_hack "${swig_output_dir}" DIRECTORY)

file(TO_NATIVE_PATH "${cmake_swig_output_dir_hack}" native_swig_output_dir)
set(swig_command_line
        ${swig_command_line}
        -outdir
        \"${native_swig_output_dir}/${swig_configuration}\"
)

foreach(swig_include ${CMAKE_CURRENT_SOURCE_DIR}/../../../Source/CNTKv2LibraryDll/API;${CMAKE_CURRENT_SOURCE_DIR}/../../common)
    file(TO_NATIVE_PATH "${swig_include}" swig_include)
    set(swig_command_line
            ${swig_command_line}
            \"-I${swig_include}\"
    )
endforeach()

file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cntk_cs.i" swig_input)
set(swig_command_line
        ${swig_command_line}
        \"${swig_input}\"
)

# Invoke the command line
add_custom_command(
    COMMENT
        "Generating SWIG Proxies for C++ and C#"
    
    DEPENDS
        cntk_cs.i
        std_unordered_map.i
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/CNTKExceptionHandling.i
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/CNTKManagedCommon.i
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/CNTKValueExtend.i
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/CNTKWarnFilters.i
    
    OUTPUT 
        # The most correct output is ${swig_generated_files}. However, some cmake generators don't support output based on generator 
        # expressions. In the name of supporting the least common denominator, specify the outputs that aren't based on generator expressions.
        #
        #   ${swig_generated_files}
        ${native_swig_generated_files}

    COMMAND ${CMAKE_COMMAND} -E make_directory "${swig_output_dir}"
    COMMAND ${swig_binary} ${swig_command_line}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cntk_cs_wrap.h" "${CMAKE_CURRENT_SOURCE_DIR}/Generated"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cntk_cs_wrap.cxx" "${CMAKE_CURRENT_SOURCE_DIR}/Generated"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CMAKE_CURRENT_SOURCE_DIR}/cntk_cs_wrap.h"
    COMMAND ${CMAKE_COMMAND} -E remove -f "${CMAKE_CURRENT_SOURCE_DIR}/cntk_cs_wrap.cxx"
)

# ----------------------------------------------------------------------
# |  CS Bindings
add_library(CSharpBindings SHARED ${native_swig_generated_files})

set_target_properties(CSharpBindings PROPERTIES RUNTIME_OUTPUT_DIRECTORY x64/)
set_target_properties(CSharpBindings PROPERTIES OUTPUT_NAME ${binary_name})
set_target_properties(CSharpBindings PROPERTIES PREFIX "")

target_include_directories(CSharpBindings
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../Source/CNTKv2LibraryDll/API
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../Source/Common/Include
)

# TODO: Eventually, we should bind the this library rather than hard-coding the value
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    string(TOLOWER "${config}" lower_config)
    
    target_link_libraries(CSharpBindings
        PRIVATE ${CNTK_BUILD_LIB_DIR_HACK}/${CMAKE_STATIC_LIBRARY_PREFIX}Cntk.Core${binary_name_suffix}${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
else()
    target_link_libraries(CSharpBindings
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../x64/${config}/Cntk.Core${binary_name_suffix}${CMAKE_STATIC_LIBRARY_SUFFIX} 
    )
endif()
