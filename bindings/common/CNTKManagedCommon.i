//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKManagedCommon.i -- Common interface definitions for C# and Java.
//

%module(directors="1") CNTKLib
//%feature("autodoc", "1");

%include <stl.i>
%include <std_wstring.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_pair.i>
%include <std_shared_ptr.i>
%include <windows.i>
%include <attribute.i>
#include <exception.i>
%include "std_unordered_map.i"

#ifdef SWIGCSHARP
%include <arrays_csharp.i>
#endif

%{
    #include "CNTKLibrary.h"
    #pragma warning(disable : 4100) //unreferenced formal parameter
%}

// shared_ptr definitions
%shared_ptr(CNTK::BackPropState);
%shared_ptr(CNTK::Function);
%shared_ptr(CNTK::CompositeFunction);
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::NDShape);
%shared_ptr(CNTK::NDArrayView);
%shared_ptr(CNTK::NDMask);
%shared_ptr(std::vector<float>);

// temaplate definitions
#ifdef SWIGCSHARP
// int/bool/double/float are already enabled with SWIG_STD_VECTOR_ENHANCED in std_vector.i
SWIG_STD_VECTOR_ENHANCED(size_t)
SWIG_STD_VECTOR_ENHANCED(std::shared_ptr<CNTK::NDArrayView>)
SWIG_STD_VECTOR_ENHANCED(CNTK::Variable)
SWIG_STD_VECTOR_ENHANCED(CNTK::Axis)
SWIG_STD_VECTOR_ENHANCED(CNTK::DeviceDescriptor)
#endif //SWIGCSHARP

%template(IntVector) std::vector<int>;
%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(NDArrayViewPtrVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(BoolVector) std::vector<bool>;
#ifdef SWIGJAVA
// need to be defined before %template(DeviceDescriptorVector)
%ignore std::vector<CNTK::DeviceDescriptor>::vector(size_type);
#endif
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(UnorderedMapVariableValuePtr) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;
%template(FunctionPtrVector) std::vector<std::shared_ptr<CNTK::Function>>;

#define IGNORE_FUNCTION %rename("$ignore", %$isfunction, fullname=1)

// They are defined twice under CNTK::Internal and under CNTK namespace
%ignore CNTK::Internal::Combine;
%ignore CNTK::Internal::Where;
%ignore CNTK::Internal::Gather;
%ignore CNTK::Internal::Scatter;
%ignore CNTK::Internal::Slice;
%ignore CNTK::Internal::MaxNumCPUThreadsSet;
%ignore CNTK::Internal::CosineDistanceWithNegativeSamples;
%ignore CNTK::Internal::Convolution;

%rename(sequence_softmax) CNTK::Sequence::Softmax;
%rename(sequence_reduce_max) CNTK::Sequence::ReduceMax;
%rename(sequence_reduce_sum) CNTK::Sequence::ReduceSum;
%rename(sequence_slice) CNTK::Sequence::Slice;

IGNORE_CLASS CNTK::Internal::TensorBoardFileWriter;
// suppress SWIG warning 302: Identifier redefined.
%ignore CNTK::Internal::TensorBoardFileWriter::TensorBoardFileWriter(const std::wstring& dir, const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& modelToVisualize = nullptr);

#ifndef _MSC_VER
IGNORE_FUNCTION _wcsdup;
#endif

#ifdef SWIGJAVA
// TODO: make Java binding deal with wchar_t correctly.
IGNORE_FUNCTION CNTK::DeviceKindName;
IGNORE_FUNCTION CNTK::VariableKindName;
#endif

//use when the wrapped method returns an idiomatic type
//for non-idiomatic types, such as the default collection wrappers use RENAME_AND_MAKE_PRIVATE below
//and then write custom method in the language specific file
#if defined(SWIGCSHARP)
#define MAKE_PRIVATE(x) %csmethodmodifiers x "private"
#elif defined(SWIGJAVA)
#define MAKE_PRIVATE(x) %javamethodmodifiers x "private"
#else
#error "MAKE_PRIVATE is not defined."
#endif

%define RENAME_AND_MAKE_PRIVATE(namespace, method)
  MAKE_PRIVATE(namespace##::##method);
  %rename (_##method) namespace##::##method
%enddef

#if defined(SWIGCSHARP)
// For C#, property needs to be added as C# code. Here we just rename the corresponding C++ method and make it as private.
#define MAKE_GETTER(namespace, method) RENAME_AND_MAKE_PRIVATE(namespace, method)
#elif defined(SWIGJAVA)
// For Java, we add "get" prefix to the method name.
%define MAKE_GETTER(namespace, method)
    %rename (get ## method) namespace##::##method
%enddef
#else
#error "MAKE_GETTER is not defined."
#endif

// include common warning filters
%include "CNTKWarnFilters.i"

%rename(AreEqual) operator==;
%rename(AreNotEqual) operator!=;
%ignore operator[];

// exception handling
%include "CNTKExceptionHandling.i"

// class DeviceDescriptor
MAKE_GETTER(CNTK::DeviceDescriptor, Id);
MAKE_GETTER(CNTK::DeviceDescriptor, CPUDevice);
MAKE_GETTER(CNTK::DeviceDescriptor, Type);
RENAME_AND_MAKE_PRIVATE(CNTK::DeviceDescriptor, AllDevices);

#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::DeviceDescriptor, SetExcludedDevices);
RENAME_AND_MAKE_PRIVATE(CNTK::DeviceDescriptor, GPUDevice);
#endif

#ifdef SWIGJAVA
%rename (setExcludedDevices) CNTK::DeviceDescriptor::SetExcludedDevices;
%rename (isLocked) CNTK::DeviceDescriptor::IsLocked;
%rename (getGPUDevice) CNTK::DeviceDescriptor::GPUDevice;
%rename (useDefaultDevice) CNTK::DeviceDescriptor::UseDefaultDevice;
%rename (trySetDefaultDevice) CNTK::DeviceDescriptor::TrySetDefaultDevice;
%rename (toString) CNTK::DeviceDescriptor::AsString;
#endif

// class Axis
IGNORE_FUNCTION CNTK::Axis::DefaultDynamicAxis();
IGNORE_FUNCTION CNTK::Axis::OperandSequenceAxis();
IGNORE_FUNCTION CNTK::Axis::DefaultBatchAxis();
IGNORE_FUNCTION CNTK::Axis::AllStaticAxes();
IGNORE_FUNCTION CNTK::Axis::AllAxes();
IGNORE_FUNCTION CNTK::Axis::DefaultInputVariableDynamicAxes();
IGNORE_FUNCTION CNTK::Axis::UnknownDynamicAxes();

MAKE_GETTER(CNTK::Axis, Name);

#ifdef SWIGCSHARP
// It cannot be a property as it has a parameter.
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, StaticAxisIndex);
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, IsOrdered);
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, IsStaticAxis);
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, IsDynamicAxis);
#endif

#ifdef SWIGJAVA
MAKE_GETTER(CNTK::Axis, StaticAxisIndex);
%rename (isOrdered) CNTK::Axis::IsOrdered;
%rename (isStaticAxis) CNTK::Axis::IsStaticAxis;
%rename (isDynamicAxis) CNTK::Axis::IsDynamicAxis;
%rename (endStaticAxis) CNTK::Axis::EndStaticAxis;
%rename (toString) CNTK::Axis::AsString;
#endif

// class Function
IGNORE_FUNCTION CNTK::Function::BlockArgumentsMapping;
IGNORE_FUNCTION CNTK::GetCorrespondingOutputVariableFromClone;
IGNORE_FUNCTION CNTK::Function::RegisterUDFDeserializeCallback;
IGNORE_FUNCTION CNTK::Function::GetUDFDeserializeCallback;
IGNORE_CLASS CNTK::Internal::UDFDeserializeCallbackWrapper;
IGNORE_FUNCTION CNTK::Internal::RegisterUDFDeserializeCallbackWrapper;
IGNORE_FUNCTION CNTK::Internal::IsNativeUserFunctionRegistered;
// Ignore exposing istream to C# for now. Todo: find a good solution to map C# System.IO.Stream to std::istream.
%ignore CNTK::Function::Load(std::istream& inputStream, const DeviceDescriptor& computeDevice= DeviceDescriptor::UseDefaultDevice());

MAKE_GETTER(CNTK::Function, Name);
MAKE_GETTER(CNTK::Function, Uid);
MAKE_GETTER(CNTK::Function, RootFunction);
MAKE_GETTER(CNTK::Function, Output);
MAKE_GETTER(CNTK::Function, OpName);
MAKE_GETTER(CNTK::Function, CurrentVersion);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Inputs);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Outputs);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Arguments);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, FindAllWithName);

#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::Function, IsComposite);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, IsPrimitive);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, IsBlock);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Load);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Clone);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Evaluate);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, FindByName);
#endif // SWIGCSHARP

// Customize type mapping for modelBuffer, used by Load
#ifdef SWIGCSHARP
%typemap(ctype) (char* buffer) "char*"
%typemap(imtype) (char* buffer) "byte[]"
%typemap(cstype) (char* buffer) "byte[]"
#endif  // SWIGCSHARP

#ifdef SWIGJAVA
%rename (isComposite) CNTK::Function::IsComposite;
%rename (isPrimitive) CNTK::Function::IsPrimitive;
%rename (isBlock) CNTK::Function::IsBlock;
%rename (load) CNTK::Function::Load;
%rename (clone) CNTK::Function::Clone;
%rename (evaluate) CNTK::Function::Evaluate;
%rename (findByName) CNTK::Function::FindByName;
%rename (setName) CNTK::Function::SetName;
%rename (combine) CNTK::Function::Combine;
%rename (blockRoot) CNTK::Function::BlockRoot;
%rename (save) CNTK::Function::Save;
%rename (restore) CNTK::Function::Restore;
%rename (toString) CNTK::Function::AsString;
#endif

// Customize type mapping for modelBuffer, used by Load
// template taken from various.i
#ifdef SWIGJAVA
%typemap(jni) (char* buffer) "jbyteArray"
%typemap(jtype) (char* buffer) "byte[]"
%typemap(jstype) (char* buffer) "byte[]"
%typemap(in) (char* buffer) {
  $1 = (char *) JCALL2(GetByteArrayElements, jenv, $input, 0);
}
%typemap(argout) (char* buffer) {
  JCALL3(ReleaseByteArrayElements, jenv, $input, (jbyte *) $1, 0);
}
%typemap(javain) (char* buffer) "$javainput"
/* Prevent default freearg typemap from being used */
%typemap(freearg) (char* buffer) ""
#endif  // SWIGJAVA

// class Varaiable
#ifndef SWIGCSHARP
%ignore CNTK::Variable::Variable;
#endif
%ignore CNTK::Variable::operator FunctionPtr;
%rename ("%s") CNTK::Variable::Variable(const FunctionPtr& function);

MAKE_GETTER(CNTK::Variable, Shape);
MAKE_GETTER(CNTK::Variable, Name);
MAKE_GETTER(CNTK::Variable, Uid);
MAKE_GETTER(CNTK::Variable, Kind);
MAKE_GETTER(CNTK::Variable, Owner);
MAKE_GETTER(CNTK::Variable, DynamicAxes);

RENAME_AND_MAKE_PRIVATE(CNTK::Variable, GetHashValue);

#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsSparse);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsInput);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsOutput);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsParameter);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsConstant);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsPlaceholder);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, NeedsGradient);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, GetDataType);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, CurrentValueTimeStamp);
#endif

#ifdef SWIGJAVA
%rename (isSparse) CNTK::Variable::IsSparse;
%rename (isInput) CNTK::Variable::IsInput;
%rename (isOutput) CNTK::Variable::IsOutput;
%rename (isParameter) CNTK::Variable::IsParameter;
%rename (isConstant) CNTK::Variable::IsConstant;
%rename (isPlaceholder) CNTK::Variable::IsPlaceholder;
%rename (needsGradient) CNTK::Variable::NeedsGradient;
%rename (getDataType) CNTK::Variable::GetDataType;
%rename (toString) CNTK::Variable::AsString;
%rename (getCurrentValueTimeStamp) CNTK::Variable::CurrentValueTimeStamp;
#endif

// class NDShape
%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);
%ignore CNTK::NDShape::InferredDimension;
%ignore CNTK::NDShape::FreeDimension;

MAKE_GETTER(CNTK::NDShape, Rank);
MAKE_GETTER(CNTK::NDShape, TotalSize);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, Dimensions);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, DimensionSize);

#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, IsUnknown);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, HasInferredDimension);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, HasFreeDimension);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, HasUnboundDimension);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, SubShape);
#endif

#ifdef SWIGJAVA
%rename (isUnknown) CNTK::NDShape::IsUnknown;
%rename (hasInferredDimension) CNTK::NDShape::HasInferredDimension;
%rename (hasFreeDimension) CNTK::NDShape::HasFreeDimension;
%rename (hasUnboundDimension) CNTK::NDShape::HasUnboundDimension;
%rename (subShape) CNTK::NDShape::SubShape;
%rename (appendShape) CNTK::NDShape::AppendShape;
%rename (alias) CNTK::NDShape::Alias;
%rename (copyFrom) CNTK::NDShape::CopyFrom;
%rename (toString) CNTK::NDShape::AsString;
#endif

%extend CNTK::NDShape {
    size_t DimensionSize(size_t axisId)
    {
        return (*self)[axisId];
    }
}

// class NDMask
// Todo: add correct typemap as they might be useful in future.
IGNORE_FUNCTION CNTK::NDMask::DataBuffer;

MAKE_GETTER(CNTK::NDMask, MaskedCount);
MAKE_GETTER(CNTK::NDMask, Device);
MAKE_GETTER(CNTK::NDMask, Shape);

#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::NDMask, InvalidateSection);
RENAME_AND_MAKE_PRIVATE(CNTK::NDMask, MarkSequenceBegin);
#endif

#ifdef SWIGJAVA
%rename (invalidateSection) CNTK::NDMask::InvalidateSection;
%rename (markSequenceBegin) CNTK::NDMask::MarkSequenceBegin;
%rename (clear) CNTK::NDMask::Clear;
%rename (deepClone) CNTK::NDMask::DeepClone;
%rename (alias) CNTK::NDMask::Alias;
%rename (copyFrom) CNTK::NDMask::CopyFrom;
#endif

// class Value
MAKE_GETTER(CNTK::Value, Device);
MAKE_GETTER(CNTK::Value, Shape);
MAKE_GETTER(CNTK::Value, Data);
MAKE_GETTER(CNTK::Value, Mask);
MAKE_GETTER(CNTK::Value, MaskedCount);

// TODO: make the following methods also private in Java, after CreateBatch/CreateSequence/... methods are implemented there.
#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::Value, IsValid);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, IsSparse);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, IsReadOnly);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, Alias);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, Create);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, GetDataType);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, GetStorageFormat);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateDenseFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateDenseDouble);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateBatchFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateBatchDouble);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateSequenceFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateSequenceDouble);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateOneHotFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CreateOneHotDouble);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CopyVariableValueTo);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CopyVariableValueToFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::Value, CopyVariableValueToDouble);
#endif // SWIGCSHARP

#ifdef SWIGCSHARP
%apply int INPUT[]  { int *colStarts }
%apply int INPUT[]  { int *rowIndices }
%apply float INPUT[]  { float *nonZeroValues }
%apply double INPUT[]  { double *nonZeroValues }
%apply int OUTPUT[]  { int *sequenceLength }
%apply int OUTPUT[]  { int *numNonZeroValues }
#endif

#ifdef SWIGJAVA
%rename (isValid) CNTK::Value::IsValid;
%rename (isSparse) CNTK::Value::IsSparse;
%rename (isReadOnly) CNTK::Value::IsReadOnly;
%rename (alias) CNTK::Value::Alias;
%rename (create) CNTK::Value::Create;
%rename (getDataType) CNTK::Value::GetDataType;
%rename (getStorageFormat) CNTK::Value::GetStorageFormat;
%rename (deepClone) CNTK::Value::DeepClone;
%rename (copyFrom) CNTK::Value::CopyFrom;
%rename (erase) CNTK::Value::Erase;
%rename (createDenseFloat) CNTK::Value::CreateDenseFloat;
%rename (createDenseDouble) CNTK::Value::CreateDenseDouble;
%rename (createBatchFloat) CNTK::Value::CreateBatchFloat;
%rename (createBatchDouble) CNTK::Value::CreateBatchDouble;
%rename (createSequenceFloat) CNTK::Value::CreateSequenceFloat;
%rename (createSequenceDouble) CNTK::Value::CreateSequenceDouble;
%rename (createOneHotFloat) CNTK::Value::CreateOneHotFloat;
%rename (createOneHotDouble) CNTK::Value::CreateOneHotDouble;
%rename (copyVariableValueTo) CNTK::Value::CopyVariableValueTo;
%rename (copyVariableValueToFloat) CNTK::Value::CopyVariableValueToFloat;
%rename (copyVariableValueToDouble) CNTK::Value::CopyVariableValueToDouble;
%rename (toString) CNTK::Value::AsString;

// TODO: make Java binding deal with double*, float * and int * correctly.
%ignore CNTK::Value::CreateSequenceFloat(const CNTK::NDShape& sampleShape, size_t sequenceLength, const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false);
%ignore CNTK::Value::CreateSequenceDouble(const CNTK::NDShape& sampleShape, size_t sequenceLength, const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false);
%ignore CNTK::Value::CreateSequenceFloat(size_t dimension, size_t sequenceLength, const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false);
%ignore CNTK::Value::CreateSequenceDouble(size_t dimension, size_t sequenceLength, const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false);
#endif // SWIGJAVA

%include "CNTKValueExtend.i"

// class NDArrayView
%ignore CNTK::NDArrayView::NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly = false);
%ignore CNTK::NDArrayView::NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device);
%ignore CNTK::NDArrayView::NDArrayView(double value, DataType dataType = DataType::Float, const NDShape& viewShape = { 1 }, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), bool readOnly = false);

MAKE_GETTER(CNTK::NDArrayView, Device);
MAKE_GETTER(CNTK::NDArrayView, Shape);

#ifdef SWIGCSHARP
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, IsSparse);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, IsReadOnly);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, Alias);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, SliceView);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, GetDataType);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, GetStorageFormat);
#endif

#ifdef SWIGJAVA
%rename (isSparse) CNTK::NDArrayView::IsSparse;
%rename (isReadOnly) CNTK::NDArrayView::IsReadOnly;
%rename (alias) CNTK::NDArrayView::Alias;
%rename (sliceView) CNTK::NDArrayView::SliceView;
%rename (getDataType) CNTK::NDArrayView::GetDataType;
%rename (getStorageFormat) CNTK::NDArrayView::GetStorageFormat;
%rename (setValue) CNTK::NDArrayView::SetValue;
%rename (deepClone) CNTK::NDArrayView::DeepClone;
%rename (asShape) CNTK::NDArrayView::AsShape;
%rename (copyFrom) CNTK::NDArrayView::CopyFrom;
%rename (changeDevice) CNTK::NDArrayView::ChangeDevice;
%rename (toString) CNTK::NDArrayView::AsString;
#endif

#ifdef SWIGCSHARP
// define typemap for dataBuffer
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }

// TODO: make Java correctly deal with float*, double* and int *
%extend CNTK::NDArrayView {
    NDArrayView(const NDShape& viewShape, float *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        if (device.Type() == CNTK::DeviceKind::GPU)
        {
            auto cpuView = new CNTK::NDArrayView(viewShape, dataBuffer, numBufferElements, CNTK::DeviceDescriptor::CPUDevice(), readOnly);
            auto gpuView = new CNTK::NDArrayView(cpuView->GetDataType(), cpuView->GetStorageFormat(), viewShape, device);
            gpuView->CopyFrom(*cpuView);
            return gpuView;
        }
        else
            return new CNTK::NDArrayView(viewShape, dataBuffer, numBufferElements, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, double *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        if (device.Type() == CNTK::DeviceKind::GPU)
        {
            auto cpuView = new CNTK::NDArrayView(viewShape, dataBuffer, numBufferElements, CNTK::DeviceDescriptor::CPUDevice(), readOnly);
            auto gpuView = new CNTK::NDArrayView(cpuView->GetDataType(), cpuView->GetStorageFormat(), viewShape, device);
            gpuView->CopyFrom(*cpuView);
            return gpuView;
        }
        else
            return new CNTK::NDArrayView(viewShape, dataBuffer, numBufferElements, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(viewShape, colStarts, rowIndices, nonZeroValues, numNonZeroValues, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(viewShape, colStarts, rowIndices, nonZeroValues, numNonZeroValues, device, readOnly);
    }
}
#endif
