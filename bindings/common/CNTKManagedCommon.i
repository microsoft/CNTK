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
    #pragma warning(disable : 4100)
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
// bool/double/float are already enabled with SWIG_STD_VECTOR_ENHANCED in std_vector.i
SWIG_STD_VECTOR_ENHANCED(size_t)
SWIG_STD_VECTOR_ENHANCED(std::shared_ptr<CNTK::NDArrayView>)
SWIG_STD_VECTOR_ENHANCED(CNTK::Variable)
SWIG_STD_VECTOR_ENHANCED(CNTK::Axis)
SWIG_STD_VECTOR_ENHANCED(CNTK::DeviceDescriptor)
#endif //SWIGCSHARP

%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(NDArrayViewPtrVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(BoolVector) std::vector<bool>;

#ifdef SWIGJAVA
%ignore std::vector<CNTK::DeviceDescriptor>::vector(size_type);
#endif
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;

%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(UnorderedMapVariableValuePtr) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;
%template(FunctionPtrVector) std::vector<std::shared_ptr<CNTK::Function>>;

// ignore items not needed.

#define %ignore_function %rename("$ignore", %$isfunction, fullname=1)
#define %ignore_class %rename("$ignore", %$isclass, fullname=1)
#define %ignore_namespace %rename("$ignore", %$isnamespace, fullname=1)
#define %ignore_variable %rename("$ignore", %$isvariable, fullname=1)
// It seems that SWIG does not understand %$isstruct.
#define %ignore_struct %rename("$ignore", fullname=1)
#define %ignore_enum_class %rename("$ignore", fullname=1)

%ignore_function CNTK::PlaceholderVariable;
%ignore_function CNTK::InputVariable;
%ignore_function CNTK::OutputVariable;

%ignore_class CNTK::Variable::CompositeFunction;
%ignore_class CNTK::Variable::Trainer;
%ignore_class CNTK::Varaiable::PrimitiveFunction;

%ignore_class CNTK::IDictionarySerializable;
%ignore_class CNTK::DictionaryValue;
// To suppress SWIG warning 302: Identifier redefined.
%ignore CNTK::DictionaryValue::operator=;
%ignore CNTK::DictionaryValue::Value;
%ignore_class CNTK::Dictionary;
%ignore_class CNTK::ParameterInitializer;

%ignore CNTK::SentinelValueForAutoSelectRandomSeed;
%ignore CNTK::SentinelValueForInferParamInitRank;
%ignore CNTK::DefaultParamInitScale;
%ignore CNTK::DefaultParamInitOutputRank;
%ignore CNTK::DefaultParamInitFilterRank;
%ignore CNTK::TimesNoInferredInputRank;
%ignore CNTK::TimesReduceSequenceAxisWithoutInferredInputRank;
%ignore_function CNTK::ConstantInitializer;
%ignore_function CNTK::UniformInitializer;
%ignore_function CNTK::NormalInitializer;
%ignore_function CNTK::XavierInitializer;
%ignore_function CNTK::GlorotUniformInitializer;
%ignore_function CNTK::GlorotNormalInitializer;
%ignore_function CNTK::HeUniformInitializer;
%ignore_function CNTK::HeNormalInitializer;
%ignore_function CNTK::BilinearInitializer;
%ignore_function CNTK::RandomInitializerWithRank;
%ignore_function CNTK::TruncatedNormalInitializer;

%ignore std::hash<CNTK::Parameter>;
%ignore_struct std::hash<::CNTK::Constant>;
%ignore_struct std::hash<::CNTK::Axis>;
%ignore_struct std::hash<::CNTK::NDShape>;
%ignore_struct std::hash<::CNTK::Variable>;

%ignore_function CNTK::Value::UnpackVariableValue;

%ignore_class CNTK::Function::CompositeFunction;
%ignore_class CNTK::Function::Trainer;

%ignore_function CNTK::Function::Backward;
%ignore_function CNTK::Function::Forward;
%ignore_function CNTK::Function::Serialize;
%ignore_function CNTK::Function::Deserialize;
%ignore_function CNTK::Function::Parameters;
%ignore_function CNTK::Function::Constants;
%ignore_function CNTK::Function::Placeholders;
%ignore_function CNTK::Function::Attributes;
%ignore_function CNTK::Function::PrintGraph;
%ignore_function CNTK::Function::BlockArgumentsMapping;
%ignore_function CNTK::Function::ReplacePlaceholders;
%ignore_function CNTK::Function::ReplacePlaceholder;
%ignore_function CNTK::Function::Function;
%ignore_function CNTK::Function::RestoreFromCheckpoint;
%ignore_function CNTK::Function::Gradients;
%ignore_function CNTK::Function::RegisterNativeUserFunction;
%ignore_function CNTK::Function::NativeUserFunction;

%ignore_class CNTK::Parameter;
%ignore_class CNTK::Constant;
%ignore_class CNTK::BackPropState;
%ignore_enum_class CNTK::PoolingType;

%ignore_function CNTK::Negate;
%ignore_function CNTK::operator-;
%ignore_function CNTK::Sigmoid;
%ignore_function CNTK::Tanh;
%ignore_function CNTK::Sin;
%ignore_function CNTK::Cos;
%ignore_function CNTK::ReLU;
%ignore_function CNTK::Exp;
%ignore_function CNTK::Log;
%ignore_function CNTK::Square;
%ignore_function CNTK::Sqrt;
%ignore_function CNTK::Round;
%ignore_function CNTK::Floor;
%ignore_function CNTK::Ceil;
%ignore_function CNTK::Abs;
%ignore_function CNTK::Reciprocal;
%ignore_function CNTK::Softmax;
%ignore_function CNTK::Hardmax;
%ignore_function CNTK::TransposeAxes;
%ignore_function CNTK::Transpose;
%ignore_function CNTK::Slice;
%ignore_function CNTK::RandomSample;
%ignore_function CNTK::RandomSampleInclusionFrequency;
%ignore_function CNTK::Dropout;
%ignore_function CNTK::Reshape;
%ignore_function CNTK::Plus;
%ignore_function CNTK::operator+;
%ignore_function CNTK::Minus;
%ignore_function CNTK::operator-;
%ignore_function CNTK::LogAddExp;
%ignore_function CNTK::Pow;
%ignore_function CNTK::ElementTimes;
%ignore_function CNTK::ElementDivide;
%ignore_function CNTK::Equal;
%ignore_function CNTK::NotEqual;
%ignore_function CNTK::Less;
%ignore_function CNTK::LessEqual;
%ignore_function CNTK::Greater;
%ignore_function CNTK::GreaterEqual;
%ignore_function CNTK::Times;
%ignore_function CNTK::TransposeTimes;
%ignore_function CNTK::CosineDistance;
%ignore_function CNTK::CosineDistanceWithNegativeSamples;
%ignore_function CNTK::BinaryCrossEntropy;
%ignore_function CNTK::WeightedBinaryCrossEntropy;
%ignore_function CNTK::SquaredError;
%ignore_function CNTK::CrossEntropyWithSoftmax;
%ignore_function CNTK::EditDistanceError;
%ignore_function CNTK::ForwardBackward;
%ignore_function CNTK::LabelsToGraph;
%ignore_function CNTK::ClassificationError;
%ignore_function CNTK::PastValue;
%ignore_function CNTK::FutureValue;
%ignore_function CNTK::ReduceSum;
%ignore_function CNTK::ReduceLogSum;
%ignore_function CNTK::ReduceMean;
%ignore_function CNTK::ReduceMax;
%ignore_function CNTK::ReduceMin;
%ignore_function CNTK::ReduceProd;
%ignore_function CNTK::PerDimMeanVarianceNormalize;
%ignore_function CNTK::Convolution;
%ignore_function CNTK::ROIPooling;
%ignore_function CNTK::ConvolutionTranspose;
%ignore_function CNTK::Pooling;
%ignore_function CNTK::Unpooling;
%ignore_function CNTK::LambdaRank;
%ignore_function CNTK::NDCGAt1;
%ignore_function CNTK::BatchNormalization;
%ignore_function CNTK::OptimizedRNNStack;
%ignore_function CNTK::Clip;
%ignore_function CNTK::ElementSelect;
%ignore_function CNTK::Splice;
%ignore_function CNTK::StopGradient;
%ignore_function CNTK::Assign;
%ignore_function CNTK::ELU;
%ignore_function CNTK::LeakyReLU;
%ignore_function CNTK::PReLU;
%ignore_function CNTK::Softplus;
%ignore_function CNTK::Argmax;
%ignore_function CNTK::Argmin;
%ignore_function CNTK::ToSequence;
%ignore_function CNTK::ToSequenceLike;
%ignore_function CNTK::AsBlock;
%ignore_function CNTK::ReaderCrop;
%ignore_function CNTK::ReaderMean;
%ignore_function CNTK::ReaderScale;
%ignore_function CNTK::ReaderColor;
%ignore_function CNTK::ImageDeserializer;
%ignore_function CNTK::Base64ImageDeserializer;
%ignore_function CNTK::CTFDeserializer;
%ignore_function CNTK::HTKFeatureDeserializer;
%ignore_function CNTK::HTKMLFDeserializer;

%ignore_namespace CNTK::Sequence;

%ignore_class CNTK::TrainingParameterSchedule;
%ignore_class CNTK::TrainingParameterPerUnitSchedule;
%ignore_class CNTK::TrainingParameterPerSampleSchedule;
%ignore_class CNTK::TrainingParameterPerMinibatchSchedule;
%ignore_class CNTK::LearningRatePerSampleSchedule;
%ignore_class CNTK::LearningRatePerMinibatchSchedule;
%ignore_class CNTK::MinibatchSizeSchedule;
%ignore_class CNTK::LearningRateSchedule;
%ignore_class CNTK::MomentumSchedule;
%ignore_class CNTK::MomentumPerSampleSchedule;
%ignore_class CNTK::MomentumPerMinibatchSchedule;
%ignore_class CNTK::MomentumAsTimeConstantSchedule;
%ignore_struct CNTK::AdditionalLearningOptions;
%ignore_class CNTK::Learner;

%ignore_function CNTK::DefaultUnitGainValue;
%ignore_function CNTK::SetDefaultUnitGainValue;

%ignore_function CNTK::SGDLearner;
%ignore_function CNTK::MomentumSGDLearner;
%ignore_function CNTK::NesterovLearner;

%ignore_variable CNTK::DefaultVarianceMomentum;

%ignore_function CNTK::FSAdaGradLearner;
%ignore_function CNTK::AdamLearner;
%ignore_function CNTK::AdaGradLearner;
%ignore_function CNTK::RMSPropLearner;
%ignore_function CNTK::AdaDeltaLearner;
%ignore_function CNTK::UniversalLearner;
%ignore_function CNTK::Internal::UniversalLearner;

%ignore_class CNTK::DistributedLearner;

%ignore_function CNTK::CreateDataParallelDistributedLearner;
%ignore_function CNTK::CreateQuantizedDataParallelDistributedLearner;
%ignore_function CNTK::CreateBlockMomentumDistributedLearner;

%ignore_class CNTK::Trainer;
%ignore_function CNTK::CreateTrainer;
%ignore_class CNTK::Evaluator;
%ignore_function CNTK::CreateEvaluator;
%ignore_struct CNTK::StreamInformation;
%ignore_struct std::hash<::CNTK::StreamInformation>;
%ignore operator==(const StreamInformation& left, const StreamInformation& right);

%ignore_struct CNTK::MinibatchData;
%ignore_class CNTK::MinibatchSource;
%ignore_struct CNTK::MinibatchInfo;
%ignore_struct CNTK::MinibatchSourceConfig;

%ignore_function CNTK::CreateCompositeMinibatchSource;
%ignore_struct CNTK::StreamConfiguration;
%ignore_struct CNTK::HTKFeatureConfiguration;
%ignore_function CNTK::TextFormatMinibatchSource;
%ignore_function CNTK::ComputeInputPerDimMeansAndInvStdDevs;
%ignore_struct CNTK::DistributedWorkerDescriptor;
%ignore operator==(const DistributedWorkerDescriptor& left, const DistributedWorkerDescriptor& right);
%ignore_class CNTK::DistributedCommunicator;
%ignore_class CNTK::QuantizedDistributedCommunicator;
%ignore_function CNTK::MPICommunicator;
%ignore_function CNTK::QuantizedMPICommunicator;
%ignore_struct CNTK::CrossValidationConfig;
%ignore_struct CNTK::CheckpointConfig;
%ignore_struct CNTK::TestConfig;

%ignore_class CNTK::TrainingSession;
%ignore_function CNTK::CreateBasicTrainingSession;
%ignore_function CNTK::CreateTrainingSession;
%ignore_function CNTK::CreateDataParallelDistributedTrainer;
%ignore_function CNTK::CreateQuantizedDataParallelDistributedTrainer;

%ignore_class CNTK::ProgressWriter;

%ignore_function CNTK::SetCheckedMode;
%ignore_function CNTK::GetCheckedMode;

%ignore_struct std::hash<::CNTK::DistributedWorkerDescriptor>;

// Ignore things in CNTKLibraryInternals.h that are not exposed for C# Eval.
%ignore_function CNTK::Internal::GenerateUid;
%ignore_enum_class CNTK::Internal::PrimitiveFunction;
%ignore_class CNTK::Internal::CompositeFunction;
%ignore_function CNTK::Internal::MaxNumCPUThreadsSet;
%ignore_enum_class CNTK::PrimitiveOpType;
%ignore_function CNTK::Internal::IsWithin;
%ignore_function CNTK::Internal::PackedIndex;
%ignore_function CNTK::Internal::GatherPacked;
%ignore_function CNTK::Internal::ScatterPacked;
%ignore_function CNTK::Internal::ReconcileDynamicAxes;
%ignore_function CNTK::Internal::ZeroesWithDynamicAxesLike;
%ignore_function CNTK::Internal::Where;
%ignore_function CNTK::Internal::Gather;
%ignore_function CNTK::Internal::Scatter;
%ignore_function CNTK::Internal::Slice;
%ignore_function CNTK::Internal::ReduceElements;
%ignore_function CNTK::Internal::CosineDistanceWithNegativeSamples;
%ignore_function CNTK::Internal::Convolution;
%ignore_function CNTK::Internal::SaveAsLegacyModel;
%ignore_function CNTK::Internal::AddProgressWriters;
%ignore_function CNTK::Internal::NewUniqueId;

%ignore_function CNTK::Internal::EnableReversingTensorShapesInErrorMessages;
%ignore_function CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled;
%ignore_function CNTK::Internal::AlwaysAllowSettingDefaultDevice;
%ignore_function CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed;
%ignore_function CNTK::Internal::AllowRenamingFunctions;
%ignore_function CNTK::Internal::IsRenamingFunctionsAllowed;
%ignore_function CNTK::Internal::SetAutomaticUnpackingOfPackedValues;
%ignore_function CNTK::Internal::IsAutomaticUnpackingOfPackedValuesDisabled;
%ignore_function CNTK::Internal::SetComputationNetworkTraceLevel;
%ignore_function CNTK::Internal::GetComputationNetworkTraceLevel;
%ignore_function CNTK::Internal::SetGPUMemoryAllocationTraceLevel;
%ignore_function CNTK::Internal::ForceSynchronousCUDAKernelExecutions;
%ignore_function CNTK::Internal::ForceDeterministicAlgorithms;
%ignore_function CNTK::Internal::ShouldForceDeterministicAlgorithms;
%ignore_function CNTK::Internal::EnableSynchronousGPUKernelExecution;
%ignore_function CNTK::Internal::IsSynchronousGPUKernelExecutionEnabled;
%ignore_function CNTK::Internal::SetFixedRandomSeed;
%ignore_function CNTK::Internal::EnableForwardValuesSharing;
%ignore_function CNTK::Internal::DisableForwardValuesSharing;
%ignore_function CNTK::Internal::EnableGradientAccumulationOptimization;
%ignore_function CNTK::Internal::DisableGradientAccumulationOptimization;
%ignore CNTK::Internal::DefaultProfilerBufferSize;
%ignore_function CNTK::Internal::StartProfiler;
%ignore_function CNTK::Internal::StopProfiler;
%ignore_function CNTK::Internal::EnableProfiler;
%ignore_function CNTK::Internal::DisableProfiler;
%ignore_function CNTK::Internal::AreEquivalent;
%ignore_function CNTK::Internal::AreEqual;
%ignore_function CNTK::Internal::PrintBuiltInfo;
%ignore_function CNTK::Internal::PrintGpuInfo;
%ignore_function CNTK::Internal::DefaultPackThresholdSizeInBytes;
%ignore_function CNTK::Internal::ToDictionary;

%ignore_class CNTK::Internal::TensorBoardFileWriter;
// Suppress SWIG warning 302: Identifier redefined.
%ignore CNTK::Internal::TensorBoardFileWriter::TensorBoardFileWriter(const std::wstring& dir, const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& modelToVisualize = nullptr);

%ignore_struct CNTK::GPUProperties;
%ignore_function CNTK::DeviceDescriptor::GetGPUProperties;

%include "CNTKWarnFilters.i"

#ifdef SWIGCSHARP
// define typemap for dataBuffer
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }
#endif

// exception handling
%include "CNTKExceptionHandling.i"

// class DeviceDescriptor
%rename (GetAllDevices) CNTK::DeviceDescriptor::AllDevices;
%rename (GetCPUDevice) CNTK::DeviceDescriptor::CPUDevice;
%rename (GetDeviceType) CNTK::DeviceDescriptor::Type;
%rename (GetId) CNTK::DeviceDescriptor::Id;
%rename (_SetExcludedDevices) CNTK::DeviceDescriptor::SetExcludedDevices;

// class Axis
%rename (GetName) CNTK::Axis::Name;
%rename (IsOrderedAxis) CNTK::Axis::IsOrdered;
%ignore_function CNTK::Axis::DefaultDynamicAxis();
%ignore_function CNTK::Axis::OperandSequenceAxis();
%ignore_function CNTK::Axis::DefaultBatchAxis();
%ignore_function CNTK::Axis::AllStaticAxes();
%ignore_function CNTK::Axis::AllAxes();
%ignore_function CNTK::Axis::DefaultInputVariableDynamicAxes();
%ignore_function CNTK::Axis::UnknownDynamicAxes();

%rename(AreEqual) operator==;
%rename(AreNotEqual) operator!=;
%ignore operator[];

// class Function
%ignore CNTK::Function::BlockArgumentsMapping;
%ignore CNTK::GetCorrespondingOutputVariableFromClone;
%rename (GetName) CNTK::Function::Name;
%rename (GetUid) CNTK::Function::Uid;
%rename (GetRootFunction) CNTK::Function::RootFunction;
%rename (GetInputs) CNTK::Function::Inputs;
%rename (GetOutput) CNTK::Function::Output;
%rename (GetOutputs) CNTK::Function::Outputs;
%rename (GetArguments) CNTK::Function::Arguments;
%rename (GetOpName) CNTK::Function::OpName;
%rename (_Clone) CNTK::Function::Clone;
%rename (_FindAllWithName) CNTK::Function::FindAllWithName;
%rename (_IsComposite) CNTK::Function::IsComposite;
%rename (_IsPrimitive) CNTK::Function::IsPrimitive;
%rename (_IsBlock) CNTK::Function::IsBlock;

%ignore CNTK::Function::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
%ignore CNTK::Function::Load(const char* buffer, size_t length, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());
// Ignore exposing istream to C# for now. Todo: find a good solution to map C# System.IO.Stream to std::istream.
%ignore CNTK::Function::Load(std::istream& inputStream, const DeviceDescriptor& computeDevice= DeviceDescriptor::UseDefaultDevice());

%extend CNTK::Function {
    static FunctionPtr Load(const std::wstring& filepath,
                            const CNTK::DeviceDescriptor& computeDevice = CNTK::DeviceDescriptor::UseDefaultDevice())
    {
        return CNTK::Function::Load(filepath, computeDevice);
    }

    static FunctionPtr Load(const char* modelBuffer, size_t length,
                            const CNTK::DeviceDescriptor& computeDevice = CNTK::DeviceDescriptor::UseDefaultDevice())
    {
        return CNTK::Function::Load(modelBuffer, length, computeDevice);
    }
}

%ignore CNTK::Function::RegisterUDFDeserializeCallback;
%ignore CNTK::Function::GetUDFDeserializeCallback;
%ignore_class CNTK::Internal::UDFDeserializeCallbackWrapper;
%ignore_function CNTK::Internal::RegisterUDFDeserializeCallbackWrapper;
%ignore_function CNTK::Internal::IsNativeUserFunctionRegistered;

#ifdef SWIGCSHARP

// Customize type mapping for modelBuffer, used by Load
%typemap(ctype) (char* modelBuffer) "char*"
%typemap(imtype) (char* modelBuffer) "byte[]"
%typemap(cstype) (char* modelBuffer) "byte[]"

#endif  // SWIGCSHARP

#ifdef SWIGJAVA

// Customize type mapping for modelBuffer, used by Load
// template taken from various.i
%typemap(jni) (char* modelBuffer) "jbyteArray"
%typemap(jtype) (char* modelBuffer) "byte[]"
%typemap(jstype) (char* modelBuffer) "byte[]"
%typemap(in) (char* modelBuffer) {
  $1 = (char *) JCALL2(GetByteArrayElements, jenv, $input, 0);
}
%typemap(argout) (char* modelBuffer) {
  JCALL3(ReleaseByteArrayElements, jenv, $input, (jbyte *) $1, 0);
}
%typemap(javain) (char* modelBuffer) "$javainput"
/* Prevent default freearg typemap from being used */
%typemap(freearg) (char* modelBuffer) ""

#endif  // SWIGJAVA

// class Varaiable
%ignore CNTK::Variable::Variable;
%ignore CNTK::Variable::operator FunctionPtr;
%rename ("%s") CNTK::Variable::Variable(const FunctionPtr& function);
%rename (GetShape) CNTK::Variable::Shape;
%rename (GetName) CNTK::Variable::Name;
%rename (GetVariableKind) CNTK::Variable::Kind;
%rename (GetDynamicAxes) CNTK::Variable::DynamicAxes;
%rename (_IsSparse) CNTK::Variable::IsSparse;
%rename (_IsInput) CNTK::Variable::IsInput;
%rename (_IsOutput) CNTK::Variable::IsOutput;
%rename (_IsParameter) CNTK::Variable::IsParameter;
%rename (_IsConstant) CNTK::Variable::IsConstant;
%rename (_IsPlaceholder) CNTK::Variable::IsPlaceholder;
%rename (GetOwner) CNTK::Variable::Owner;

// class NDShape
%rename (GetDimensions) CNTK::NDShape::Dimensions;
%rename (GetRank) CNTK::NDShape::Rank;
%rename (GetTotalSize) CNTK::NDShape::TotalSize;
%rename (_IsUnknown) CNTK::NDShape::IsUnknown;
%rename (_HasInferredDimension) CNTK::NDShape::HasInferredDimension;
%rename (_HasFreeDimension) CNTK::NDShape::HasFreeDimension;

%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);
%ignore CNTK::NDShape::InferredDimension;
%ignore CNTK::NDShape::FreeDimension;

%extend CNTK::NDShape {
    size_t GetDimensionSize(size_t axisId)
    {
        return (*self)[axisId];
    }
}

// class NDMask
// Todo: add correct typemap as they might be useful in future.
%ignore_function CNTK::NDMask::DataBuffer;
%rename (GetMaskedCount) CNTK::NDMask::MaskedCount;
%rename (GetDevice) CNTK::NDMask::Device;
%rename (GetShape) CNTK::NDMask::Shape;
%rename (_InvalidateSection) CNTK::NDMask::InvalidateSection;
%rename (_MarkSequenceBegin) CNTK::NDMask::MarkSequenceBegin;
%rename (_InvalidateSection) CNTK::NDMask::InvalidateSection;

#ifdef SWIGCSHARP
// class Value
%apply int INPUT[]  { int *colStarts }
%apply int INPUT[]  { int *rowIndices }
%apply float INPUT[]  { float *nonZeroValues }
%apply double INPUT[]  { double *nonZeroValues }
#endif

%rename (GetDevice) CNTK::Value::Device;
%rename (GetShape) CNTK::Value::Shape;
%rename (_IsSparse) CNTK::Value::IsSparse;
%rename (_IsReadOnly) CNTK::Value::IsReadOnly;
%rename (_MaskedCount) CNTK::Value::MaskedCount;

%include "CNTKValueExtend.i"

// class NDArrayView
%ignore CNTK::NDArrayView::NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly = false);
%ignore CNTK::NDArrayView::NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device);
%ignore CNTK::NDArrayView::NDArrayView(double value, DataType dataType = DataType::Float, const NDShape& viewShape = { 1 }, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), bool readOnly = false);

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

%rename (GetDevice) CNTK::NDArrayView::Device;
%rename (GetShape) CNTK::NDArrayView::Shape;
%rename (_IsSparse) CNTK::NDArrayView::IsSparse;
%rename (_IsReadOnly) CNTK::NDArrayView::IsReadOnly;
%rename (_SliceView) CNTK::NDArrayView::SliceView;
