%module(directors="1") Utils
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

%{
    #include "CNTKLibrary.h"
    #pragma warning(disable : 4100)
%}

%shared_ptr(CNTK::BackPropState);
%shared_ptr(CNTK::Function);
%shared_ptr(CNTK::CompositeFunction);
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::NDShape);
%shared_ptr(CNTK::NDArrayView);
%shared_ptr(CNTK::NDMask);
%shared_ptr(std::vector<float>);

%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;

%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(NDArrayViewPtrVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(BoolVector) std::vector<bool>;
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;

%template(UnorderedMapVariableValuePtr) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;
%template(FunctionPtrVector) std::vector<std::shared_ptr<CNTK::Function>>;

%template() std::vector<bool>;
%template() std::vector<int>;
%template() std::pair<size_t, double>;
%template() std::vector<std::pair<size_t, double>>;

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

%ignore_struct std::hash<::CNTK::Parameter>;
%ignore_struct CNTK::hash<::CNTK::Constant>;

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
%ignore_function CNTK::ELU;
%ignore_function CNTK::LeakyReLU;
%ignore_function CNTK::PReLU;
%ignore_function CNTK::Softplus;
%ignore_function CNTK::Argmax;
%ignore_function CNTK::Argmin;
%ignore_function CNTK::AsBlock;
%ignore_function CNTK::ReaderCrop;
%ignore_function CNTK::ReaderMean;
%ignore_function CNTK::ReaderScale;
%ignore_function CNTK::ReaderColor;
%ignore_function CNTK::ImageDeserializer;
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
%ignore_function CNTK::Internal::ReconcileDynamicAxis;
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
%ignore_function CNTK::Internal::SetComputationNetworkTrackGapNans;
%ignore_function CNTK::Internal::GetComputationNetworkTrackGapNans;
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
%ignore_function CNTK::PrintBuiltInfo;
%ignore_function CNTK::Internal::DefaultPackThresholdSizeInBytes;
%ignore_function CNTK::Internal::ToDictionary;

%ignore_class CNTK::Internal::TensorBoardFileWriter;

%ignore_struct CNTK::GPUProperties;
%ignore_function CNTK::DeviceDescriptor::GetGPUProperties;

// map the pointer to array
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }

// Exception handling
%include "CNTK_ExceptionHandling.i"
