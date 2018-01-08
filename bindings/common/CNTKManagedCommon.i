//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKManagedCommon.i -- Common interface definitions for C Sharp and Java.
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

// include common warning filters
%include "CNTKWarnFilters.i"

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
// For C Sharp, property needs to be added as C Sharp code. Here we just rename the corresponding C++ method and make it as private.
#define MAKE_GETTER(namespace, method) RENAME_AND_MAKE_PRIVATE(namespace, method)
#elif defined(SWIGJAVA)
// For Java, we add "get" prefix to the method name.
%define MAKE_GETTER(namespace, method)
    %rename (get ## method) namespace##::##method
%enddef
#else
#error "MAKE_GETTER is not defined."
#endif

#ifdef SWIGCSHARP
// make swig generated classes partial in order to put shim layer classes as partial of it.
%pragma(csharp) moduleclassmodifiers="public partial class"
%typemap(csclassmodifiers) CNTK::DeviceDescriptor "public partial class"
%typemap(csclassmodifiers) CNTK::Axis "public partial class"
%typemap(csclassmodifiers) CNTK::Function "public partial class"
%typemap(csclassmodifiers) CNTK::NDShape "public partial class"
%typemap(csclassmodifiers) CNTK::NDMask "public partial class"
%typemap(csclassmodifiers) CNTK::Variable "public partial class"
%typemap(csclassmodifiers) CNTK::Parameter "public partial class"
%typemap(csclassmodifiers) CNTK::Constant "public partial class"
%typemap(csclassmodifiers) CNTK::Value "public partial class"
%typemap(csclassmodifiers) CNTK::NDArrayView "public partial class"
%typemap(csclassmodifiers) CNTK::StreamConfiguration "public partial class"
%typemap(csclassmodifiers) CNTK::Trainer "public partial class"
%typemap(csclassmodifiers) CNTK::Learner "public partial class"
%typemap(csclassmodifiers) CNTK::MinibatchSource "public partial class"
%typemap(csclassmodifiers) CNTK::MinibatchSourceConfig "public partial class"
#endif

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
// unsigned char/char/int/bool/double/float are already enabled with SWIG_STD_VECTOR_ENHANCED in std_vector.i
SWIG_STD_VECTOR_ENHANCED(size_t)
SWIG_STD_VECTOR_ENHANCED(std::shared_ptr<CNTK::NDArrayView>)
SWIG_STD_VECTOR_ENHANCED(CNTK::Variable)
SWIG_STD_VECTOR_ENHANCED(CNTK::Axis)
SWIG_STD_VECTOR_ENHANCED(CNTK::DeviceDescriptor)
SWIG_STD_VECTOR_ENHANCED(CNTK::Dictionary)
SWIG_STD_VECTOR_ENHANCED(CNTK::Parameter)
SWIG_STD_VECTOR_ENHANCED(CNTK::ProgressWriter)
SWIG_STD_VECTOR_ENHANCED(CNTK::Learner)
#endif //SWIGCSHARP

%template(CharVector) std::vector<char>;
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

#ifdef SWIGCSHARP
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::ProgressWriter);
%shared_ptr(CNTK::Learner);
%shared_ptr(CNTK::DistributedLearner);
%shared_ptr(CNTK::Trainer);
%shared_ptr(CNTK::MinibatchSource);
%shared_ptr(CNTK::Evaluator);
%template(UnsignedCharVector) std::vector<unsigned char>;
%template(DictionaryVector) std::vector<CNTK::Dictionary>;
%template (UnorderedMapStreamInformationMinibatchData) std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData>;
%template(UnorderedMapVariableMinibatchData) std::unordered_map<CNTK::Variable, CNTK::MinibatchData>;
%template(StreamConfigurationVector) std::vector<CNTK::StreamConfiguration>;
%template(ParameterVector) std::vector<CNTK::Parameter>;
%template(ConstantVector) std::vector<CNTK::Constant>;
%template(StringVector) std::vector<std::wstring>;
%template(HTKFeatureConfigurationVector) std::vector<CNTK::HTKFeatureConfiguration>;
%template(UnorderedMapParameterNDArrayViewPtr) std::unordered_map<CNTK::Parameter, std::shared_ptr<CNTK::NDArrayView>>;
%template(PairNDArrayViewPtrNDArrayViewPtr) std::pair<std::shared_ptr<CNTK::NDArrayView>, std::shared_ptr<CNTK::NDArrayView>>;
%template(UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr) std::unordered_map<CNTK::StreamInformation, std::pair<std::shared_ptr<CNTK::NDArrayView>, std::shared_ptr<CNTK::NDArrayView>>>;
%template(ProgressWriterVector) std::vector<std::shared_ptr<CNTK::ProgressWriter>>;
%template(LearnerVector) std::vector<std::shared_ptr<CNTK::Learner>>;
%template(UnorderedMapStringDictionaryValue) std::unordered_map<std::wstring, CNTK::DictionaryValue>;
%template(PairSizeTDouble) std::pair<size_t, double>;
%template(VectorPairSizeTDouble) std::vector<std::pair<size_t, double>>;
%template(PairSizeTSizeT) std::pair<size_t, size_t>;
%template(PairSizeTInt) std::pair<size_t, int>;
%template(PairIntInt) std::pair<int, int>;
%template(PairFloatFloat) std::pair<float, float>;
%template(PairDoubleDouble) std::pair<double, double>;
#endif

// ignore items not needed.
#define IGNORE_FUNCTION %rename("$ignore", %$isfunction, fullname=1)
#define IGNORE_CLASS %rename("$ignore", %$isclass, fullname=1)
#define IGNORE_NAMESPACE %rename("$ignore", %$isnamespace, fullname=1)
#define IGNORE_VARIABLE %rename("$ignore", %$isvariable, fullname=1)
// It seems that SWIG does not understand %$isstruct.
#define IGNORE_STRUCT %rename("$ignore", fullname=1)
#define IGNORE_ENUM_CLASS %rename("$ignore", fullname=1)

#ifndef _MSC_VER
IGNORE_FUNCTION _wcsdup;
#endif

IGNORE_CLASS CNTK::Variable::CompositeFunction;
IGNORE_CLASS CNTK::Varaiable::PrimitiveFunction;
IGNORE_CLASS CNTK::IDictionarySerializable;
// To suppress SWIG warning 302: Identifier redefined.
%ignore CNTK::DictionaryValue::operator=;
%ignore CNTK::DictionaryValue::Value;
%ignore CNTK::NDArrayView::AdjustSparseBlockColumn;

IGNORE_CLASS CNTK::ParameterInitializer;

%ignore CNTK::SentinelValueForAutoSelectRandomSeed;
%ignore CNTK::DefaultParamInitOutputRank;
%ignore CNTK::DefaultParamInitFilterRank;
%ignore CNTK::TimesNoInferredInputRank;
%ignore CNTK::TimesReduceSequenceAxisWithoutInferredInputRank;
IGNORE_STRUCT std::hash<CNTK::Parameter>;
IGNORE_STRUCT std::hash<::CNTK::Constant>;
IGNORE_STRUCT std::hash<::CNTK::Axis>;
IGNORE_STRUCT std::hash<::CNTK::NDShape>;
IGNORE_STRUCT std::hash<::CNTK::Variable>;
IGNORE_FUNCTION CNTK::Value::UnpackVariableValue;
IGNORE_CLASS CNTK::Function::CompositeFunction;
IGNORE_CLASS CNTK::Function::Trainer;
IGNORE_FUNCTION CNTK::Function::Backward;
IGNORE_FUNCTION CNTK::Function::Forward;
IGNORE_FUNCTION CNTK::Function::Serialize;
IGNORE_FUNCTION CNTK::Function::Deserialize;
IGNORE_FUNCTION CNTK::Function::BlockArgumentsMapping;
IGNORE_FUNCTION CNTK::Function::Function;
IGNORE_FUNCTION CNTK::Function::RestoreFromCheckpoint;
IGNORE_FUNCTION CNTK::Function::Gradients;
IGNORE_FUNCTION CNTK::Function::RegisterNativeUserFunction;
IGNORE_FUNCTION CNTK::Function::NativeUserFunction;
IGNORE_FUNCTION CNTK::Function::SetAttribute;
IGNORE_CLASS CNTK::BackPropState;
IGNORE_FUNCTION CNTK::operator+;
IGNORE_FUNCTION CNTK::operator-;
IGNORE_FUNCTION CNTK::AsBlock;
IGNORE_FUNCTION CNTK::NCELoss;


#ifndef SWIGCSHARP
IGNORE_CLASS CNTK::TrainingParameterSchedule;
#else
%ignore CNTK::TrainingParameterSchedule::TrainingParameterSchedule(TrainingParameterSchedule<T>&&); 
%ignore CNTK::TrainingParameterSchedule::operator=;
%ignore CNTK::TrainingParameterSchedule::Transform;
#endif

IGNORE_CLASS CNTK::TrainingParameterPerMinibatchSchedule;
IGNORE_CLASS CNTK::MinibatchSizeSchedule;
IGNORE_CLASS CNTK::LearningRateSchedule;
IGNORE_CLASS CNTK::MomentumSchedule;
IGNORE_FUNCTION CNTK::DefaultUnitGainValue;
IGNORE_FUNCTION CNTK::SetDefaultUnitGainValue;
IGNORE_FUNCTION CNTK::NesterovLearner;
IGNORE_VARIABLE CNTK::DefaultVarianceMomentum;

IGNORE_FUNCTION CNTK::Learner::GetOptions;

IGNORE_FUNCTION CNTK::UniversalLearner;
IGNORE_FUNCTION CNTK::Internal::UniversalLearner;
IGNORE_CLASS CNTK::DistributedLearner;
IGNORE_FUNCTION CNTK::CreateDataParallelDistributedLearner;
IGNORE_FUNCTION CNTK::CreateQuantizedDataParallelDistributedLearner;
IGNORE_FUNCTION CNTK::CreateBlockMomentumDistributedLearner;
IGNORE_STRUCT std::hash<::CNTK::StreamInformation>;
%ignore operator==(const StreamInformation& left, const StreamInformation& right);
IGNORE_STRUCT CNTK::DistributedWorkerDescriptor;
%ignore operator==(const DistributedWorkerDescriptor& left, const DistributedWorkerDescriptor& right);
IGNORE_CLASS CNTK::DistributedCommunicator;
IGNORE_CLASS CNTK::QuantizedDistributedCommunicator;
IGNORE_FUNCTION CNTK::MPICommunicator;
IGNORE_FUNCTION CNTK::QuantizedMPICommunicator;
IGNORE_STRUCT CNTK::CrossValidationConfig;
IGNORE_STRUCT CNTK::CheckpointConfig;
IGNORE_STRUCT CNTK::TestConfig;
IGNORE_CLASS CNTK::TrainingSession;
IGNORE_FUNCTION CNTK::CreateBasicTrainingSession;
IGNORE_FUNCTION CNTK::CreateTrainingSession;
IGNORE_FUNCTION CNTK::CreateDataParallelDistributedTrainer;
IGNORE_FUNCTION CNTK::CreateQuantizedDataParallelDistributedTrainer;
IGNORE_FUNCTION CNTK::SetCheckedMode;
IGNORE_FUNCTION CNTK::GetCheckedMode;
IGNORE_STRUCT std::hash<::CNTK::DistributedWorkerDescriptor>;
IGNORE_FUNCTION CNTK::Internal::GenerateUid;
IGNORE_ENUM_CLASS CNTK::Internal::PrimitiveFunction;
IGNORE_CLASS CNTK::Internal::CompositeFunction;
IGNORE_FUNCTION CNTK::Internal::MaxNumCPUThreadsSet;
IGNORE_ENUM_CLASS CNTK::PrimitiveOpType;
IGNORE_FUNCTION CNTK::Internal::IsWithin;
IGNORE_FUNCTION CNTK::Internal::PackedIndex;
IGNORE_FUNCTION CNTK::Internal::GatherPacked;
IGNORE_FUNCTION CNTK::Internal::ScatterPacked;
IGNORE_FUNCTION CNTK::Internal::ReconcileDynamicAxes;
IGNORE_FUNCTION CNTK::Internal::ZeroesWithDynamicAxesLike;
IGNORE_FUNCTION CNTK::Internal::Where;
IGNORE_FUNCTION CNTK::Internal::Gather;
IGNORE_FUNCTION CNTK::Internal::Scatter;
IGNORE_FUNCTION CNTK::Internal::Slice;
IGNORE_FUNCTION CNTK::Internal::ReduceElements;
IGNORE_FUNCTION CNTK::Internal::CosineDistanceWithNegativeSamples;
IGNORE_FUNCTION CNTK::Internal::Convolution;
IGNORE_FUNCTION CNTK::Internal::SaveAsLegacyModel;
IGNORE_FUNCTION CNTK::Internal::AddProgressWriters;
IGNORE_FUNCTION CNTK::Internal::NewUniqueId;
IGNORE_FUNCTION CNTK::Internal::EnableReversingTensorShapesInErrorMessages;
IGNORE_FUNCTION CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled;
IGNORE_FUNCTION CNTK::Internal::AlwaysAllowSettingDefaultDevice;
IGNORE_FUNCTION CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed;
IGNORE_FUNCTION CNTK::Internal::AllowRenamingFunctions;
IGNORE_FUNCTION CNTK::Internal::IsRenamingFunctionsAllowed;
IGNORE_FUNCTION CNTK::Internal::SetAutomaticUnpackingOfPackedValues;
IGNORE_FUNCTION CNTK::Internal::IsAutomaticUnpackingOfPackedValuesDisabled;
IGNORE_FUNCTION CNTK::Internal::SetComputationNetworkTraceLevel;
IGNORE_FUNCTION CNTK::Internal::GetComputationNetworkTraceLevel;
IGNORE_FUNCTION CNTK::Internal::SetGPUMemoryAllocationTraceLevel;
IGNORE_FUNCTION CNTK::Internal::ForceSynchronousCUDAKernelExecutions;
IGNORE_FUNCTION CNTK::Internal::ForceDeterministicAlgorithms;
IGNORE_FUNCTION CNTK::Internal::ShouldForceDeterministicAlgorithms;
IGNORE_FUNCTION CNTK::Internal::EnableSynchronousGPUKernelExecution;
IGNORE_FUNCTION CNTK::Internal::IsSynchronousGPUKernelExecutionEnabled;
IGNORE_FUNCTION CNTK::Internal::SetFixedRandomSeed;
IGNORE_FUNCTION CNTK::Internal::EnableForwardValuesSharing;
IGNORE_FUNCTION CNTK::Internal::DisableForwardValuesSharing;
IGNORE_FUNCTION CNTK::Internal::EnableGradientAccumulationOptimization;
IGNORE_FUNCTION CNTK::Internal::DisableGradientAccumulationOptimization;
%ignore CNTK::Internal::DefaultProfilerBufferSize;
IGNORE_FUNCTION CNTK::Internal::StartProfiler;
IGNORE_FUNCTION CNTK::Internal::StopProfiler;
IGNORE_FUNCTION CNTK::Internal::EnableProfiler;
IGNORE_FUNCTION CNTK::Internal::DisableProfiler;
IGNORE_FUNCTION CNTK::Internal::AreEquivalent;
IGNORE_FUNCTION CNTK::Internal::AreEqual;
IGNORE_FUNCTION CNTK::Internal::PrintBuiltInfo;
IGNORE_FUNCTION CNTK::Internal::PrintGpuInfo;
IGNORE_FUNCTION CNTK::Internal::DefaultPackThresholdSizeInBytes;
IGNORE_FUNCTION CNTK::Internal::ToDictionary;
IGNORE_CLASS CNTK::Internal::TensorBoardFileWriter;
// suppress SWIG warning 302: Identifier redefined.
%ignore CNTK::Internal::TensorBoardFileWriter::TensorBoardFileWriter(const std::wstring& dir, const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& modelToVisualize = nullptr);

IGNORE_STRUCT CNTK::GPUProperties;
IGNORE_FUNCTION CNTK::DeviceDescriptor::GetGPUProperties;

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
MAKE_GETTER(CNTK::Axis, Name);

// class Function
IGNORE_FUNCTION CNTK::Function::BlockArgumentsMapping;
IGNORE_FUNCTION CNTK::GetCorrespondingOutputVariableFromClone;
IGNORE_FUNCTION CNTK::Function::RegisterUDFDeserializeCallback;
IGNORE_FUNCTION CNTK::Function::GetUDFDeserializeCallback;
IGNORE_CLASS CNTK::Internal::UDFDeserializeCallbackWrapper;
IGNORE_FUNCTION CNTK::Internal::RegisterUDFDeserializeCallbackWrapper;
IGNORE_FUNCTION CNTK::Internal::IsNativeUserFunctionRegistered;
// Ignore exposing istream to C Sharp for now. Todo: find a good solution to map C Sharp System.IO.Stream to std::istream.
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

%rename ("%s") CNTK::Variable::Variable(const FunctionPtr& function);

MAKE_GETTER(CNTK::Variable, Shape);
MAKE_GETTER(CNTK::Variable, Name);
MAKE_GETTER(CNTK::Variable, Uid);
MAKE_GETTER(CNTK::Variable, Kind);
MAKE_GETTER(CNTK::Variable, Owner);
MAKE_GETTER(CNTK::Variable, DynamicAxes);

RENAME_AND_MAKE_PRIVATE(CNTK::Variable, GetHashValue);

// class NDShape
%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);
%ignore CNTK::NDShape::InferredDimension;
%ignore CNTK::NDShape::FreeDimension;

MAKE_GETTER(CNTK::NDShape, Rank);
MAKE_GETTER(CNTK::NDShape, TotalSize);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, Dimensions);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, DimensionSize);

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

// class Value
MAKE_GETTER(CNTK::Value, Device);
MAKE_GETTER(CNTK::Value, Shape);
MAKE_GETTER(CNTK::Value, Data);
MAKE_GETTER(CNTK::Value, Mask);
MAKE_GETTER(CNTK::Value, MaskedCount);

// class NDArrayView
%ignore CNTK::NDArrayView::NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly = false);
%ignore CNTK::NDArrayView::NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device);
%ignore CNTK::NDArrayView::NDArrayView(double value, DataType dataType = DataType::Float, const NDShape& viewShape = { 1 }, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), bool readOnly = false);

MAKE_GETTER(CNTK::NDArrayView, Device);
MAKE_GETTER(CNTK::NDArrayView, Shape);

#ifdef SWIGJAVA
IGNORE_CLASS CNTK::DictionaryValue;
IGNORE_CLASS CNTK::Dictionary;
IGNORE_FUNCTION CNTK::PlaceholderVariable;
IGNORE_FUNCTION CNTK::InputVariable;
IGNORE_FUNCTION CNTK::OutputVariable;
IGNORE_CLASS CNTK::Variable::Trainer;
%ignore CNTK::SentinelValueForInferParamInitRank;
%ignore CNTK::DefaultParamInitScale;
IGNORE_FUNCTION CNTK::ConstantInitializer;
IGNORE_FUNCTION CNTK::GlorotUniformInitializer;
IGNORE_FUNCTION CNTK::UniformInitializer;
IGNORE_FUNCTION CNTK::NormalInitializer;
IGNORE_FUNCTION CNTK::XavierInitializer;
IGNORE_FUNCTION CNTK::GlorotNormalInitializer;
IGNORE_FUNCTION CNTK::HeUniformInitializer;
IGNORE_FUNCTION CNTK::HeNormalInitializer;
IGNORE_FUNCTION CNTK::BilinearInitializer;
IGNORE_FUNCTION CNTK::RandomInitializerWithRank;
IGNORE_FUNCTION CNTK::TruncatedNormalInitializer;
IGNORE_FUNCTION CNTK::Function::Parameters;
IGNORE_FUNCTION CNTK::Function::ReplacePlaceholders;
IGNORE_FUNCTION CNTK::Function::ReplacePlaceholder;
IGNORE_FUNCTION CNTK::Function::Placeholders;
IGNORE_FUNCTION CNTK::Function::PrintGraph;
IGNORE_FUNCTION CNTK::Function::Constants;
IGNORE_FUNCTION CNTK::Function::Attributes;
IGNORE_CLASS CNTK::Parameter;
IGNORE_CLASS CNTK::Constant;
IGNORE_ENUM_CLASS CNTK::PoolingType;
IGNORE_FUNCTION CNTK::Negate;
IGNORE_FUNCTION CNTK::Sigmoid;
IGNORE_FUNCTION CNTK::Tanh;
IGNORE_FUNCTION CNTK::Atanh;
IGNORE_FUNCTION CNTK::Sin;
IGNORE_FUNCTION CNTK::Cos;
IGNORE_FUNCTION CNTK::Acos;
IGNORE_FUNCTION CNTK::Asin;
IGNORE_FUNCTION CNTK::Cosh;
IGNORE_FUNCTION CNTK::Sinh;
IGNORE_FUNCTION CNTK::Asinh;
IGNORE_FUNCTION CNTK::ReLU;
IGNORE_FUNCTION CNTK::Exp;
IGNORE_FUNCTION CNTK::Log;
IGNORE_FUNCTION CNTK::Square;
IGNORE_FUNCTION CNTK::Sqrt;
IGNORE_FUNCTION CNTK::Round;
IGNORE_FUNCTION CNTK::Floor;
IGNORE_FUNCTION CNTK::Ceil;
IGNORE_FUNCTION CNTK::Abs;
IGNORE_FUNCTION CNTK::Reciprocal;
IGNORE_FUNCTION CNTK::TransposeAxes;
IGNORE_FUNCTION CNTK::Transpose;
IGNORE_FUNCTION CNTK::RandomSample;
IGNORE_FUNCTION CNTK::RandomSampleInclusionFrequency;
IGNORE_FUNCTION CNTK::Dropout;
IGNORE_FUNCTION CNTK::Reshape;
IGNORE_FUNCTION CNTK::Plus;
IGNORE_FUNCTION CNTK::Minus;
IGNORE_FUNCTION CNTK::LogAddExp;
IGNORE_FUNCTION CNTK::Pow;
IGNORE_FUNCTION CNTK::ElementTimes;
IGNORE_FUNCTION CNTK::ElementDivide;
IGNORE_FUNCTION CNTK::Equal;
IGNORE_FUNCTION CNTK::NotEqual;
IGNORE_FUNCTION CNTK::Less;
IGNORE_FUNCTION CNTK::LessEqual;
IGNORE_FUNCTION CNTK::Greater;
IGNORE_FUNCTION CNTK::GreaterEqual;
IGNORE_FUNCTION CNTK::Times;
IGNORE_FUNCTION CNTK::TransposeTimes;
IGNORE_FUNCTION CNTK::CosineDistance;
IGNORE_FUNCTION CNTK::CosineDistanceWithNegativeSamples;
IGNORE_FUNCTION CNTK::BinaryCrossEntropy;
IGNORE_FUNCTION CNTK::WeightedBinaryCrossEntropy;
IGNORE_FUNCTION CNTK::SquaredError;
IGNORE_FUNCTION CNTK::CrossEntropyWithSoftmax;
IGNORE_FUNCTION CNTK::SequenceWithLattice;
IGNORE_FUNCTION CNTK::EditDistanceError;
IGNORE_FUNCTION CNTK::ForwardBackward;
IGNORE_FUNCTION CNTK::LabelsToGraph;
IGNORE_FUNCTION CNTK::ClassificationError;
IGNORE_FUNCTION CNTK::PastValue;
IGNORE_FUNCTION CNTK::FutureValue;
IGNORE_FUNCTION CNTK::Slice;
IGNORE_FUNCTION CNTK::ReduceMax;
IGNORE_FUNCTION CNTK::ReduceMin;
IGNORE_FUNCTION CNTK::Softmax;
IGNORE_FUNCTION CNTK::Hardmax;
IGNORE_FUNCTION CNTK::ReduceSum;
IGNORE_FUNCTION CNTK::ReduceLogSum;
IGNORE_FUNCTION CNTK::ReduceMean;
IGNORE_FUNCTION CNTK::ReduceProd;
IGNORE_FUNCTION CNTK::PerDimMeanVarianceNormalize;
IGNORE_FUNCTION CNTK::Convolution;
IGNORE_FUNCTION CNTK::ConvolutionTranspose;
IGNORE_FUNCTION CNTK::Pooling;
IGNORE_FUNCTION CNTK::Unpooling;
IGNORE_FUNCTION CNTK::LambdaRank;
IGNORE_FUNCTION CNTK::NDCGAt1;
IGNORE_FUNCTION CNTK::BatchNormalization;
IGNORE_FUNCTION CNTK::OptimizedRNNStack;
IGNORE_FUNCTION CNTK::Clip;
IGNORE_FUNCTION CNTK::ElementSelect;
IGNORE_FUNCTION CNTK::Splice;
IGNORE_FUNCTION CNTK::StopGradient;
IGNORE_FUNCTION CNTK::Assign;
IGNORE_FUNCTION CNTK::ELU;
IGNORE_FUNCTION CNTK::SELU;
IGNORE_FUNCTION CNTK::LeakyReLU;
IGNORE_FUNCTION CNTK::PReLU;
IGNORE_FUNCTION CNTK::Softplus;
IGNORE_FUNCTION CNTK::Argmax;
IGNORE_FUNCTION CNTK::Argmin;
IGNORE_FUNCTION CNTK::ToSequence;
IGNORE_FUNCTION CNTK::ToSequenceLike;
IGNORE_FUNCTION CNTK::ReaderMean;
IGNORE_FUNCTION CNTK::ReaderScale;
IGNORE_FUNCTION CNTK::ReaderColor;
IGNORE_NAMESPACE CNTK::Sequence;
IGNORE_FUNCTION CNTK::ROIPooling;
IGNORE_FUNCTION CNTK::ReaderCrop;
IGNORE_FUNCTION CNTK::ImageDeserializer;
IGNORE_FUNCTION CNTK::Base64ImageDeserializer;
IGNORE_FUNCTION CNTK::CTFDeserializer;
IGNORE_FUNCTION CNTK::CBFDeserializer;
IGNORE_FUNCTION CNTK::HTKFeatureDeserializer;
IGNORE_FUNCTION CNTK::HTKMLFDeserializer;
IGNORE_FUNCTION CNTK::LatticeDeserializer;
IGNORE_FUNCTION CNTK::MomentumAsTimeConstantSchedule;
IGNORE_CLASS CNTK::TrainingParameterSchedule;
IGNORE_STRUCT CNTK::AdditionalLearningOptions;
IGNORE_CLASS CNTK::Learner;
IGNORE_FUNCTION CNTK::SGDLearner;
IGNORE_FUNCTION CNTK::MomentumSGDLearner;
IGNORE_FUNCTION CNTK::FSAdaGradLearner;
IGNORE_FUNCTION CNTK::AdamLearner;
IGNORE_FUNCTION CNTK::AdaGradLearner;
IGNORE_FUNCTION CNTK::RMSPropLearner;
IGNORE_FUNCTION CNTK::AdaDeltaLearner;
IGNORE_FUNCTION CNTK::UniversalLearner;
IGNORE_CLASS CNTK::Trainer;
IGNORE_FUNCTION CNTK::CreateTrainer;
IGNORE_STRUCT CNTK::StreamInformation;
IGNORE_STRUCT CNTK::MinibatchData;
IGNORE_CLASS CNTK::MinibatchSource;
IGNORE_STRUCT CNTK::MinibatchInfo;
IGNORE_STRUCT CNTK::MinibatchSourceConfig;
IGNORE_STRUCT CNTK::StreamConfiguration;
IGNORE_FUNCTION CNTK::CreateCompositeMinibatchSource;
IGNORE_FUNCTION CNTK::TextFormatMinibatchSource;
IGNORE_STRUCT CNTK::HTKFeatureConfiguration;
IGNORE_FUNCTION CNTK::ComputeInputPerDimMeansAndInvStdDevs;
IGNORE_CLASS CNTK::ProgressWriter;
IGNORE_FUNCTION CNTK::DeviceKindName;
IGNORE_FUNCTION CNTK::VariableKindName;
IGNORE_FUNCTION CNTK::Axis::DefaultDynamicAxis();
IGNORE_FUNCTION CNTK::Axis::OperandSequenceAxis();
IGNORE_FUNCTION CNTK::Axis::DefaultBatchAxis();
IGNORE_FUNCTION CNTK::Axis::AllStaticAxes();
IGNORE_FUNCTION CNTK::Axis::AllAxes();
IGNORE_FUNCTION CNTK::Axis::DefaultInputVariableDynamicAxes();
IGNORE_FUNCTION CNTK::Axis::UnknownDynamicAxes();
%rename (setExcludedDevices) CNTK::DeviceDescriptor::SetExcludedDevices;
%rename (isLocked) CNTK::DeviceDescriptor::IsLocked;
%rename (getGPUDevice) CNTK::DeviceDescriptor::GPUDevice;
%rename (useDefaultDevice) CNTK::DeviceDescriptor::UseDefaultDevice;
%rename (trySetDefaultDevice) CNTK::DeviceDescriptor::TrySetDefaultDevice;
%rename (toString) CNTK::DeviceDescriptor::AsString;
MAKE_GETTER(CNTK::Axis, StaticAxisIndex);
%rename (isOrdered) CNTK::Axis::IsOrdered;
%rename (isStaticAxis) CNTK::Axis::IsStaticAxis;
%rename (isDynamicAxis) CNTK::Axis::IsDynamicAxis;
%rename (endStaticAxis) CNTK::Axis::EndStaticAxis;
%rename (toString) CNTK::Axis::AsString;
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
// Customize type mapping for modelBuffer, used by Load
// template taken from various.i
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
%ignore CNTK::Variable::Variable;
%ignore CNTK::Variable::operator FunctionPtr;
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
%rename (isUnknown) CNTK::NDShape::IsUnknown;                                           
%rename (hasInferredDimension) CNTK::NDShape::HasInferredDimension;
%rename (hasFreeDimension) CNTK::NDShape::HasFreeDimension;
%rename (hasUnboundDimension) CNTK::NDShape::HasUnboundDimension;
%rename (subShape) CNTK::NDShape::SubShape;
%rename (appendShape) CNTK::NDShape::AppendShape;
%rename (alias) CNTK::NDShape::Alias;
%rename (copyFrom) CNTK::NDShape::CopyFrom;
%rename (toString) CNTK::NDShape::AsString;
%rename (invalidateSection) CNTK::NDMask::InvalidateSection;
%rename (markSequenceBegin) CNTK::NDMask::MarkSequenceBegin;
%rename (clear) CNTK::NDMask::Clear;
%rename (deepClone) CNTK::NDMask::DeepClone;
%rename (alias) CNTK::NDMask::Alias;
%rename (copyFrom) CNTK::NDMask::CopyFrom;
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
%rename (isSparse) CNTK::NDArrayView::IsSparse;
%rename (isReadOnly) CNTK::NDArrayView::IsReadOnly;
%rename (isSliceView) CNTK::NDArrayView::IsSliceView;
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
IGNORE_CLASS CNTK::Evaluator;
IGNORE_FUNCTION CNTK::CreateEvaluator;
#else
%rename(SequenceIsFirst) CNTK::Sequence::IsFirst;
%rename(SequenceIsLast) CNTK::Sequence::IsLast;
%rename(SequenceSlice) CNTK::Sequence::Slice;
%rename(SequenceReduceSum) CNTK::Sequence::ReduceSum;
%rename(SequenceReduceMax) CNTK::Sequence::ReduceMax;
%rename(SequenceSoftmax) CNTK::Sequence::Softmax;
%rename(SequenceFirst) CNTK::Sequence::First;
%rename(SequenceLast) CNTK::Sequence::Last;
%rename(SequenceWhere) CNTK::Sequence::Where;
%rename(SequenceGather) CNTK::Sequence::Gather;
%rename(SequenceScatter) CNTK::Sequence::Scatter;
%rename(SequenceBroadcastAs) CNTK::Sequence::BroadcastAs;
%rename(SequenceUnpack) CNTK::Sequence::Unpack;
%ignore CNTK::InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, const wchar_t* name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore CNTK::InputVariable(const NDShape& shape, DataType dataType, const wchar_t* name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
%ignore operator>>(std::istream& stream, DictionaryValue& us);
%ignore operator<<(std::ostream& stream, const DictionaryValue& us);
%ignore CNTK::DictionaryValue::Value() const;
%ignore CNTK::DictionaryValue::DictionaryValue(DictionaryValue&& other);
%ignore CNTK::DictionaryValue::DictionaryValue(const std::vector<::CNTK::DictionaryValue>& value);
%rename(CNTKDictionary) CNTK::Dictionary;
%ignore CNTK::Dictionary::Dictionary(Dictionary&& other);
%ignore CNTK::Dictionary::operator=(Dictionary&& other);
IGNORE_FUNCTION CNTK::AddConfigString;
IGNORE_FUNCTION CNTK::Dictionary::Contains(const std::wstring& key) const;
%rename(Equal) CNTK::Dictionary::operator=;
%ignore operator>>(std::istream& stream, Dictionary& dictionary);
%ignore operator<<(std::ostream& stream, const Dictionary& dictionary);
IGNORE_FUNCTION CNTK::Dictionary::begin() const;
IGNORE_FUNCTION CNTK::Dictionary::cbegin() const;
IGNORE_FUNCTION CNTK::Dictionary::end() const;
IGNORE_FUNCTION CNTK::Dictionary::cend() const;
IGNORE_FUNCTION CNTK::Dictionary::Keys();
%ignore CNTK::TrainingParameterSchedule::TrainingParameterSchedule(TrainingParameterSchedule<T>&&); 
%ignore CNTK::TrainingParameterSchedule::operator=;
%ignore CNTK::AdditionalLearningOptions::gaussianNoiseInjectionStdDev;
IGNORE_FUNCTION CNTK::UniversalLearner(const std::vector<Parameter>& parameters, const ParameterUpdateFunctor& func);
IGNORE_FUNCTION CNTK::Trainer::ParameterLearners;
%ignore CNTK::MinibatchSource::StreamInfos;
%ignore CNTK::MinibatchSource::InfinitelyRepeat;
%ignore CNTK::MinibatchSource::FullDataSweep;
%ignore CNTK::MinibatchSource::DefaultRandomizationWindowInChunks;
%ignore CNTK::MinibatchSourceConfig::isMultithreaded;
// RENAME_AND_MAKE_PRIVATE(CNTK::MinibatchSource, GetNextMinibatch);
IGNORE_FUNCTION CNTK::ProgressWriter::UpdateTraining;
IGNORE_FUNCTION CNTK::ProgressWriter::UpdateTest;
IGNORE_FUNCTION CNTK::ProgressWriter::UpdateDistributedSync;
IGNORE_FUNCTION CNTK::ProgressWriter::WriteTrainingSummary;
IGNORE_FUNCTION CNTK::ProgressWriter::WriteTestSummary;
RENAME_AND_MAKE_PRIVATE(CNTK::DeviceDescriptor, SetExcludedDevices);
RENAME_AND_MAKE_PRIVATE(CNTK::DeviceDescriptor, GPUDevice);
// It cannot be a property as it has a parameter.
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, StaticAxisIndex);
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, IsOrdered);
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, IsStaticAxis);
RENAME_AND_MAKE_PRIVATE(CNTK::Axis, IsDynamicAxis);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Parameters);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, IsComposite);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, IsPrimitive);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, IsBlock);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Load);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Save);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Clone);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, Evaluate);
RENAME_AND_MAKE_PRIVATE(CNTK::Function, FindByName);
RENAME_AND_MAKE_PRIVATE(CNTK::Trainer, TrainMinibatch);
// Customize type mapping for modelBuffer, used by Load
%typemap(ctype) (char* buffer) "char*"
%typemap(imtype) (char* buffer) "byte[]"
%typemap(cstype) (char* buffer) "byte[]"
%rename(ToFunction) CNTK::Variable::operator FunctionPtr;
// TODO: make the following methods also private in Java, after CreateBatch/CreateSequence/... methods are implemented there.
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsSparse);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsInput);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsOutput);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsParameter);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsConstant);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, IsPlaceholder);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, NeedsGradient);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, GetDataType);
RENAME_AND_MAKE_PRIVATE(CNTK::Variable, CurrentValueTimeStamp);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, IsUnknown);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, HasInferredDimension);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, HasFreeDimension);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, HasUnboundDimension);
RENAME_AND_MAKE_PRIVATE(CNTK::NDShape, SubShape);
RENAME_AND_MAKE_PRIVATE(CNTK::NDMask, InvalidateSection);
RENAME_AND_MAKE_PRIVATE(CNTK::NDMask, MarkSequenceBegin);
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
RENAME_AND_MAKE_PRIVATE(CNTK::Constant, ScalarFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::Constant, ScalarDouble);
%apply int INPUT[]  { int *colStarts }
%apply int INPUT[]  { int *rowIndices }
%apply float INPUT[]  { float *nonZeroValues }
%apply double INPUT[]  { double *nonZeroValues }
%apply int OUTPUT[]  { int *sequenceLength }
%apply int OUTPUT[]  { int *numNonZeroValues }
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, IsSparse);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, IsReadOnly);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, IsSliceView);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, Alias);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, SliceView);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, GetDataType);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, GetStorageFormat);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, RandomNormalFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, RandomNormalDouble);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, RandomUniformFloat);
RENAME_AND_MAKE_PRIVATE(CNTK::NDArrayView, RandomUniformDouble);
// define typemap for dataBuffer
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }

// ProgressWriters returns unordered_set which is not supportted by swig CCharp
IGNORE_FUNCTION CNTK::Evaluator::ProgressWriters;
#endif

%include "CNTKValueExtend.i"
