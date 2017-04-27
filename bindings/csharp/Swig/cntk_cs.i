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
%include <arrays_csharp.i>
#include <exception.i>

// include the unordered_map.i.
%include "std_unordered_map.i"

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

SWIG_STD_VECTOR_ENHANCED(size_t)
%template(SizeTVector) std::vector<size_t>;
SWIG_STD_VECTOR_ENHANCED(double)
%template(DoubleVector) std::vector<double>;
SWIG_STD_VECTOR_ENHANCED(float)
%template(FloatVector) std::vector<float>;
SWIG_STD_VECTOR_ENHANCED(CNTK::Variable)
%template(VariableVector) std::vector<CNTK::Variable>;
SWIG_STD_VECTOR_ENHANCED(CNTK::Axis)
%template(AxisVector) std::vector<CNTK::Axis>;
SWIG_STD_VECTOR_ENHANCED(std::shared_ptr<CNTK::NDArrayView>)
%template(NDArrayViewPtrVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
SWIG_STD_VECTOR_ENHANCED(bool)
%template(BoolVector) std::vector<bool>;
SWIG_STD_VECTOR_ENHANCED(CNTK::DeviceDescriptor)
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

%rename (GetAllDevices) CNTK::DeviceDescriptor::AllDevices;
%rename (GetCPUDevice) CNTK::DeviceDescriptor::CPUDevice;
%rename (GetDeviceType) CNTK::DeviceDescriptor::Type;
%rename (GetId) CNTK::DeviceDescriptor::Id;
%rename (_SetExcludedDevices) CNTK::DeviceDescriptor::SetExcludedDevices;
%rename (AreEqualDeviceDescriptor) CNTK::operator==(const DeviceDescriptor& left, const DeviceDescriptor& right);

%typemap(cscode) CNTK::DeviceDescriptor %{

    public int Id
    {
        get { return (int)GetId(); }
    }

    public DeviceKind Type
    {
        get { return GetDeviceType(); }
    }

    public static DeviceDescriptor CPUDevice
    {
        get { return GetCPUDevice(); }
    }

    public static DeviceDescriptor GPUDevice(int deviceId)
    {
        if (deviceId < 0)
        {
            throw new System.ArgumentException("The paraemter deviceId should not be a negative value");
        }
        return GPUDevice((uint)deviceId);
    }

    public static System.Collections.Generic.IList<DeviceDescriptor> AllDevices()
    {
        var deviceVector = GetAllDevices();
        // The CopyTo is to ensure the elements in the deviceVector can live beyond deviceVector itself.
        var deviceArray = new DeviceDescriptor[deviceVector.Count];
        deviceVector.CopyTo(deviceArray);
        var deviceList = new System.Collections.Generic.List<DeviceDescriptor>(deviceArray);
        return deviceList;
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        DeviceDescriptor p = obj as DeviceDescriptor;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualDeviceDescriptor(this, p);
    }

    public bool Equals(DeviceDescriptor p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualDeviceDescriptor(this, p);
    }

    public static bool operator ==(DeviceDescriptor first, DeviceDescriptor second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualDeviceDescriptor(first, second);
    }

    public static bool operator !=(DeviceDescriptor first, DeviceDescriptor second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        return this.GetDeviceType().GetHashCode();
    }

    public static void SetExcludedDevices(System.Collections.Generic.IEnumerable<DeviceDescriptor> excluded)
    {
        var excludeVector = new DeviceDescriptorVector();
        foreach (var element in excluded)
        {
            excludeVector.Add(element);
        }
        _SetExcludedDevices(excludeVector);
    }
%}

%rename (GetName) CNTK::Axis::Name;
%rename (IsOrderedAxis) CNTK::Axis::IsOrdered;
%rename (AreEqualAxis) CNTK::operator==(const Axis& first, const Axis& second);
%ignore_function CNTK::Axis::DefaultDynamicAxis();
%ignore_function CNTK::Axis::OperandSequenceAxis();
%ignore_function CNTK::Axis::DefaultBatchAxis();
%ignore_function CNTK::Axis::AllStaticAxes();
%ignore_function CNTK::Axis::AllAxes();
%ignore_function CNTK::Axis::DefaultInputVariableDynamicAxes();
%ignore_function CNTK::Axis::UnknownDynamicAxes();

%typemap(cscode) CNTK::Axis %{

    public string Name
    {
        get 
        {
            return GetName();
        }
    }

    public bool IsStatic
    {
        get 
        {
            return IsStaticAxis();
        }
    }

    public bool IsDynamic
    {
        get 
        {
            return IsDynamicAxis();
        }
    }

    public bool IsOrdered
    {
        get 
        {
            return IsOrderedAxis();
        }
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        Axis p = obj as Axis;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualAxis(this, p);
    }

    public bool Equals(Axis p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualAxis(this, p);
    }

    public static bool operator ==(Axis first, Axis second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualAxis(first, second);
    }

    public static bool operator !=(Axis first, Axis second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        if (this.IsDynamicAxis())
        {
            return this.GetName().GetHashCode();
        }
        else
        {
            return this.StaticAxisIndex().GetHashCode();
        }
    }
%}

%ignore CNTK::Function::BlockArgumentsMapping;
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

// Customize type mapping for modelBuffer, used by Load
%apply char* INPUT { char* modelBuffer }
%typemap(ctype) (char* modelBuffer) "char*"
%typemap(imtype) (char* modelBuffer) "byte[]"
%typemap(cstype) (char* modelBuffer) "byte[]"

%typemap(cscode) CNTK::Function %{

    public static Function Load(byte[] modelBuffer, DeviceDescriptor computeDevice)
    {
        return Load(modelBuffer, (uint)modelBuffer.Length, computeDevice);
    }

    public string Name
    {
        get 
        {
            return GetName();
        }
    }

    public string Uid
    {
        get 
        {
            return GetUid();
        }
    }

    public Function RootFunction
    {
        get { return GetRootFunction(); }
    }

    public System.Collections.Generic.IList<Variable> Outputs
    {
        get {
            var varVector = GetOutputs();
            var varArray = new Variable[varVector.Count];
            // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
            varVector.CopyTo(varArray);
            var varList = new System.Collections.Generic.List<Variable>(varArray);
            return varList;
        }
    }

    public Variable Output
    {
        get { return GetOutput(); }
    }

    public string OpName
    {
        get { return GetOpName(); }
    }

    public bool IsComposite
    {
        get { return _IsComposite(); }
    }

    public bool IsPrimitive
    {
        get { return _IsPrimitive(); }
    }

    public bool IsBlock
    {
        get { return _IsBlock(); }
    }

    public System.Collections.Generic.IList<Variable> Arguments
    {
        get {
            var varVector = GetArguments();
            var varArray = new Variable[varVector.Count];
            // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
            varVector.CopyTo(varArray);
            var varList = new System.Collections.Generic.List<Variable>(varArray);
            return varList;
        }
    }

    public System.Collections.Generic.IList<Variable> Inputs
    {
        get {
            var varVector = GetInputs();
            var varArray = new Variable[varVector.Count];
            // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
            varVector.CopyTo(varArray);
            var varList = new System.Collections.Generic.List<Variable>(varArray);
            return varList;
        }
    }

    public static Function Combine(System.Collections.Generic.IEnumerable<Variable> operands)
    {
        var varVect = new VariableVector();
        foreach (var v in operands)
        {
            varVect.Add(v);
        }
        return CNTKLib.Combine(varVect);
    }

    public static Function AsComposite(Function rootFunction, string name = "")
    {
        return CNTKLib.AsComposite(rootFunction, name);
    }

    public static Function Alias(Variable operand, string name = "")
    {
        return CNTKLib.Alias(operand, name);
    }

    // For C# Eval, default ParameterCloningMethod is share.
    public Function Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod.Share)
    {
        return _Clone(ParameterCloningMethod.Share);
    }

    public void Evaluate(System.Collections.Generic.IDictionary<Variable, Value> inputs, System.Collections.Generic.IDictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
    {
        Evaluate(inputs, outputs, false, computeDevice);
    }

    public void Evaluate(System.Collections.Generic.IDictionary<Variable, Value> inputs, System.Collections.Generic.IDictionary<Variable, Value> outputs, bool createPersistentOutputValues, DeviceDescriptor computeDevice)
    {
        // Evaluate the rootFunction.
        var inMap = new UnorderedMapVariableValuePtr();
        var outMap = new UnorderedMapVariableValuePtr();
        foreach (var p in inputs)
        {
            inMap.Add(p.Key, p.Value);
        }

        foreach (var p in outputs)
        {
            outMap.Add(p.Key, p.Value);
        }

        Evaluate(inMap, outMap, computeDevice);

        foreach (var p in outMap)
        {
            if (createPersistentOutputValues && (outputs[p.Key] == null))
            {
                outputs[p.Key] = p.Value.DeepClone();
            }
            else
            { 
                // for shared_ptr<Value>, the p.Value returns a copy, so it is safe to use it directly in outputs.
                outputs[p.Key] = p.Value;
            }
        }
    }

    public System.Collections.Generic.IList<Function> FindAllWithName(string name, bool nestedSearchInsideBlockFunction = false)
    {
        var funcPtrVector = _FindAllWithName(name, nestedSearchInsideBlockFunction);
        var funcPtrList = new System.Collections.Generic.List<Function>(funcPtrVector.Count);
        for (int i = 0; i < funcPtrVector.Count; i++)
        {
            // for shared_ptr, the funcPtrVector[i] returns a copy, so it is safe to directly use it in return list.
            funcPtrList.Add(funcPtrVector[i]);
        }
        return funcPtrList;
    }
%}

%ignore CNTK::Variable::Variable;
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
%rename (AreEqualVariable) CNTK::operator==(const Variable& first, const Variable& second);

%typemap(cscode) CNTK::Variable %{

    public NDShape Shape
    {
        get { return GetShape(); }
    }

    public string Name
    {
        get { return GetName(); }
    }

    public VariableKind Kind
    {
        get { return GetVariableKind(); }
    }

    public DataType DataType
    {
        get { return GetDataType(); }
    }

    public System.Collections.Generic.IList<Axis> DynamicAxes
    {
        get {
            var axisVector = GetDynamicAxes();
            // The CopyTo is to ensure that elements in axisVector live beyond the lifecycle of axisVector.
            var axisArray = new Axis[axisVector.Count];
            axisVector.CopyTo(axisArray);
            var axisList = new System.Collections.Generic.List<Axis>(axisArray);
            return axisList;
        }
    }

    public bool IsSparse
    {
        get { return _IsSparse(); }
    }

    public bool IsInput
    {
        get { return _IsInput(); }
    }

    public bool IsOutput
    {
        get { return _IsOutput(); }
    }

    public bool IsParameter
    {
        get { return _IsParameter(); }
    }

    public bool IsConstant
    {
        get { return _IsConstant(); }
    }

    public bool IsPlaceholder
    {
        get { return _IsPlaceholder(); }
    }

    public Function Owner
    {
        get { return GetOwner(); }
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        Variable p = obj as Variable;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualVariable(this, p);
    }

    public bool Equals(Variable p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualVariable(this, p);
    }

    public static bool operator ==(Variable first, Variable second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualVariable(first, second);
    }

    public static bool operator !=(Variable first, Variable second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        // Todo: the hash value in C++ is size_t, but only in in C#
        return (int)GetHashValue();
    }
%}

%rename (GetDimensions) CNTK::NDShape::Dimensions;
%rename (GetRank) CNTK::NDShape::Rank;
%rename (GetTotalSize) CNTK::NDShape::TotalSize;
%rename (AreEqualShape) CNTK::operator==(const NDShape& first, const NDShape& second);
%rename (_IsUnknown) CNTK::NDShape::IsUnknown;
%rename (_HasInferredDimension) CNTK::NDShape::HasInferredDimension;
%rename (_HasFreeDimension) CNTK::NDShape::HasFreeDimension;

%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);
%ignore CNTK::NDShape::InferredDimension;
%ignore CNTK::NDShape::FreeDimension;

//
// NDShape
//
%extend CNTK::NDShape {
    size_t GetDimensionSize(size_t axisId)
    {
        return (*self)[axisId];
    }
}

%typemap(cscode) CNTK::NDShape %{

    public NDShape(int numAxes, int dimension) : this((uint)numAxes, (uint)dimension)
    {
        if (numAxes < 0 || dimension < 0)
        {
            throw new System.ArgumentException("The paraemter numAxes or dimension should not be a negative value");
        }
    }

    public NDShape(int numAxes) : this((uint)numAxes)
    {
        if (numAxes < 0)
        {
            throw new System.ArgumentException("The paraemter numAxes should not be a negative value");
        }
    }

    public int Rank
    {
        get { return (int)GetRank(); }
    }

    public System.Collections.Generic.IList<int> Dimensions
    {
        get
        {
            var dimList = GetDimensions();
            var retList = new System.Collections.Generic.List<int>(dimList.Count);
            foreach (var element in dimList)
            {
                retList.Add((int)element);
            }
            return retList;
        }
    }

    public bool IsUnknown 
    {
        get { return _IsUnknown(); }
    }

    public bool HasInferredDimension
    {
        get { return _HasInferredDimension(); }
    }

    public bool HasFreeDimension
    {
        get { return _HasFreeDimension(); }
    }

    public int TotalSize
    {
        get { return (int)GetTotalSize(); }
    }

    public int this[int key]
    {
        get { return (int)GetDimensionSize((uint)key); }
    }

    public NDShape SubShape(int beginAxisId, int endAxisId)
    {
        if (beginAxisId < 0 || endAxisId < 0)
        {
            throw new System.ArgumentException("The paraemter beginAxisId or endAxisId should not be a negative value");
        }
        return SubShape((uint)beginAxisId, (uint)endAxisId);
    }

    public NDShape SubShape(int beginAxisId)
    {
        if (beginAxisId < 0)
        {
            throw new System.ArgumentException("The paraemter beginAxisId should not be a negative value");
        }
        return SubShape((uint)beginAxisId);
    }

    public static NDShape CreateNDShape(System.Collections.Generic.IEnumerable<int> dimensions)
    {
        var dimVector = new SizeTVector();
        foreach (var element in dimensions)
        {
            if (element < 0)
            {
                throw new System.ArgumentException("The paraemter diemnsions cannot contain a negative value");
            }
            dimVector.Add((uint)element);
        }
        return new NDShape(dimVector);
    }

    public override bool Equals(System.Object obj)
    {
        // If parameter is null return false.
        if (obj == null)
        {
            return false;
        }

        // If parameter cannot be cast to Point return false.
        NDShape p = obj as NDShape;
        if ((System.Object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualShape(this, p);
    }

    public bool Equals(NDShape p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualShape(this, p);
    }

    public static bool operator ==(NDShape first, NDShape second)
    {
        // If both are null, or both are same instance, return true.
        if (System.Object.ReferenceEquals(first, second))
        {
            return true;
        }

        // If one is null, but not both, return false.
        if (((object)first == null) || ((object)second == null))
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqualShape(first, second);
    }

    public static bool operator !=(NDShape first, NDShape second)
    {
        return !(first == second);
    }

    public override int GetHashCode()
    {
        //Todo: another hash function??
        return this.GetDimensions().GetHashCode();
    }

    public static readonly int InferredDimension = -1;
    public static readonly int FreeDimension = -3;
%}

// Todo: add correct typemap as they might be useful for C# in future.
%ignore_function CNTK::NDMask::DataBuffer;
%rename (GetMaskedCount) CNTK::NDMask::MaskedCount;
%rename (GetDevice) CNTK::NDMask::Device;
%rename (GetShape) CNTK::NDMask::Shape;
%rename (_InvalidateSection) CNTK::NDMask::InvalidateSection;
%rename (_MarkSequenceBegin) CNTK::NDMask::MarkSequenceBegin;
%rename (_InvalidateSection) CNTK::NDMask::InvalidateSection;

%typemap(cscode) CNTK::NDMask %{
    public void InvalidateSection(System.Collections.Generic.IEnumerable<int> sectionOffset, NDShape sectionShape) {
        var offsetVector = AsSizeTVector(sectionOffset);
        _InvalidateSection(offsetVector, sectionShape);
    }

    public void MarkSequenceBegin(System.Collections.Generic.IEnumerable<int> offset) {
        var offsetVector = AsSizeTVector(offset);
        _MarkSequenceBegin(offsetVector);
    }

    public void MarkSequenceBegin(System.Collections.Generic.IEnumerable<int> offset, NDShape sectionShape) {
        var offsetVector = AsSizeTVector(offset);
        _MarkSequenceBegin(offsetVector, sectionShape);
    }

    public int MaskedCount {
        get { return (int)GetMaskedCount(); }
    }

    public DeviceDescriptor Device {
        get { return GetDevice(); }
    }

    public NDShape Shape {
        get { return GetShape(); }
    }

    private static SizeTVector AsSizeTVector(System.Collections.Generic.IEnumerable<int> input)
    {
        var inputVector = new SizeTVector();
        foreach (var element in input)
        {
            if (element < 0)
            {
                throw new System.ArgumentException("The argument cannot contain a negative value");
            }
            inputVector.Add((uint)element);
        }
        return inputVector;
    }
%}

%apply int INPUT[]  { int *colStarts }
%apply int INPUT[]  { int *rowIndices }
%apply float INPUT[]  { float *nonZeroValues }
%apply double INPUT[]  { double *nonZeroValues }

%rename (GetDevice) CNTK::Value::Device;
%rename (GetShape) CNTK::Value::Shape;
%rename (_IsSparse) CNTK::Value::IsSparse;
%rename (_IsReadOnly) CNTK::Value::IsReadOnly;
%rename (_MaskedCount) CNTK::Value::MaskedCount;

%typemap(cscode) CNTK::Value %{

    public DeviceDescriptor Device
    {
        get
        {
            return GetDevice();
        }
    }

    public DataType DataType
    {
        get
        {
            return GetDataType();
        }
    }

    public StorageFormat StorgeFormat
    {
        get
        {
            return GetStorageFormat();
        }
    }

    public NDShape Shape
    {
        get
        {
            return GetShape();
        }
    }

    public bool IsSparse
    {
        get
        {
            return _IsSparse();
        }
    }

    public bool IsReadOnly
    {
        get
        {
            return _IsReadOnly();
        }
    }

    public int MaskedCount
    {
        get
        {
            return (int)_MaskedCount();
        }
    }

    // Create Value object from dense input: batch, sequence or batch of sequences.
    public static Value CreateBatch<T>(NDShape sampleShape, System.Collections.Generic.IEnumerable<T> batch, DeviceDescriptor device, bool readOnly = false)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            var inputVector = AsFloatVector(batch);
            return Value.CreateBatchFloat(sampleShape, inputVector, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputVector = AsDoubleVector(batch);
            return Value.CreateBatchDouble(sampleShape, inputVector, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateSequence<T>(NDShape sampleShape,
                                          System.Collections.Generic.IEnumerable<T> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(sampleShape, sequence, true, device, readOnly);
    }

    public static Value CreateSequence<T>(NDShape sampleShape,
                                          System.Collections.Generic.IEnumerable<T> sequence,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            var inputVector = AsFloatVector(sequence);
            return Value.CreateSequenceFloat(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputVector = AsDoubleVector(sequence);
            return Value.CreateSequenceDouble(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<T>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(sampleShape, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<T>> batchOfSequences,
                                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    public static Value Create<T>(NDShape sampleShape,
                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<T>> sequences,
                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = AsBoolVector(sequenceStartFlags);
        if (typeof(T).Equals(typeof(float)))
        {
            var inputAsSequencesVector = new FloatVectorVector();
            foreach (var seq in sequences)
            {
                var seqVector = AsFloatVector(seq);
                // The seqVector is copied when adding to inputAsSequencesVector.
                inputAsSequencesVector.Add(seqVector);
            }
            return Value.CreateDenseFloat(sampleShape, inputAsSequencesVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputAsSequencesVector = new DoubleVectorVector();
            foreach (var seq in sequences)
            {
                var seqVector = AsDoubleVector(seq);
                inputAsSequencesVector.Add(seqVector);
            }
            return Value.CreateDenseDouble(sampleShape, inputAsSequencesVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input, for N-dimenstional tensor. Only Create() method for now.
    public static Value Create<T>(NDShape sampleShape,
                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> sequences,
                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = AsBoolVector(sequenceStartFlags);
        var inputSeqVector = new SizeTVectorVector();
        foreach (var seq in sequences)
        {
            var s = AsSizeTVector(seq);
            inputSeqVector.Add(s);
        }
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateOneHotFloat(sampleShape, inputSeqVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateOneHotDouble(sampleShape, inputSeqVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input, for 1D tensor: batch, sequence or batch of sequences
    public static Value CreateBatch<T>(int dimension, System.Collections.Generic.IEnumerable<int> batch, DeviceDescriptor device, bool readOnly = false)
    {
        var inputVector = AsSizeTVector(batch);
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateBatchFloat((uint)dimension, inputVector, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateBatchDouble((uint)dimension, inputVector, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateSequence<T>(int dimension,
                                          System.Collections.Generic.IEnumerable<int> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(dimension, sequence, true, device, readOnly);
    }

    public static Value CreateSequence<T>(int dimension,
                                          System.Collections.Generic.IEnumerable<int> sequence,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        var inputVector = AsSizeTVector(sequence);
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateSequenceFloat((uint)dimension, inputVector, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateSequenceDouble((uint)dimension, inputVector, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateBatchOfSequences<T>(int dimension,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(int dimension,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> batchOfSequences,
                                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    public static Value Create<T>(int dimension,
                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> sequences,
                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = AsBoolVector(sequenceStartFlags);
        var inputSeqVector = new SizeTVectorVector();
        foreach (var seq in sequences)
        {
            var s = AsSizeTVector(seq);
            inputSeqVector.Add(s);
        }
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateOneHotFloat((uint)dimension, inputSeqVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateOneHotDouble((uint)dimension, inputSeqVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from sparse input, for N-dimensional tensor. Only CreateSequence() for now.
    public static Value CreateSequence<T>(NDShape sampleShape, int sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        if (nonZeroValues.Length != rowIndices.Length)
        {
            throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (colStarts.Length != sequenceLength + 1)
        {
            throw new System.ArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
        }
        uint numNonZeroValues = (uint)nonZeroValues.Length;

        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateSequenceFloat(sampleShape, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as float[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateSequenceDouble(sampleShape, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as double[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateSequence<T>(NDShape sampleShape, int sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return Value.CreateSequence<T>(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, true, device, readOnly);
    }

    // Create Value object from sparse input, for 1D tensor. Only CreateSequence() for now.
    public static Value CreateSequence<T>(int dimension, int sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        if (nonZeroValues.Length != rowIndices.Length)
        {
            throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (colStarts.Length != sequenceLength + 1)
        {
            throw new System.ArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
        }
        uint numNonZeroValues = (uint)nonZeroValues.Length;

        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateSequenceFloat((uint)dimension, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as float[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateSequenceDouble((uint)dimension, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as double[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateSequence<T>(int dimension, int sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return Value.CreateSequence<T>(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, true, device, readOnly);
    }

    // Create value object from NDArrayView
    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.IEnumerable<NDArrayView> sequences,
                               DeviceDescriptor device,
                               bool readOnly = false)
    {
        return Create(sampleShape, sequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.IEnumerable<NDArrayView> sequences,
                               System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                               DeviceDescriptor device,
                               bool readOnly = false)
    {
        var seqVector = new NDArrayViewPtrVector();
        foreach (var element in sequences)
        {
            seqVector.Add(element);
        }
        var startFlags = AsBoolVector(sequenceStartFlags);
        return Create(sampleShape, seqVector, startFlags, device, false);
    }

    //
    // Return the data of the Value object as a list of sequences with variable length.
    // This method returns an IList<IList<T>>. Each element of the outer list represents a sequence.
    // Each sequence, represented by IList<T>, contains a variable number of samples.
    // Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the variable shape.
    // The number of samples = (the count of elements in IList<T>)/(the count of elements of the sample)
    // The shape of the variable should match the shape of the Value object.
    //
    public System.Collections.Generic.IList<System.Collections.Generic.IList<T>> GetDenseData<T>(Variable outputVariable)
    {
        var sequences = new System.Collections.Generic.List<System.Collections.Generic.IList<T>>();
        if (typeof(T).Equals(typeof(float)))
        {
            if (GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new FloatVectorVector();
            CopyVariableValueToFloat(outputVariable, seqVec);

            foreach (var seq in seqVec)
            {
                var seqList = seq as System.Collections.Generic.IList<T>;
                if (seqList == null)
                    throw new System.TypeAccessException("Cannot convert to the value type.");
                // It is required to create a new List from seq, since seq is dependent on the life cycle of seqVec.
                sequences.Add(new System.Collections.Generic.List<T>(seqList));
            }
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            if (GetDataType() != DataType.Double)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new DoubleVectorVector();
            CopyVariableValueToDouble(outputVariable, seqVec);
            foreach (var seq in seqVec)
            {
                var seqList = seq as System.Collections.Generic.IList<T>;
                if (seqList == null)
                    throw new System.TypeAccessException("Cannot convert to the value type.");
                // It is required to create a new List from seq, since seq is dependent on the life cycle of seqVec.
                sequences.Add(new System.Collections.Generic.List<T>(seqList));
            }
        }
        else
        {
            throw new System.ArgumentException("The value type does not match the list type.");
        }
        return sequences;
    }

    //
    // Return the data of the Value object as a list of sequences with variable length.
    // This method returns an IList<IList<T>>. Each element of the outer list represents a sequence.
    // Each sequence, represented by List<int>, contains a variable number of samples.
    // Each sample is represented by an index of the OneHot vector. The size of the OneHot vector should match that defined in the variable.
    // The number of samples = the count of elements in List<int>.
    //
    public System.Collections.Generic.IList<System.Collections.Generic.IList<int>> GetOneHotData(Variable outputVariable)
    {
        var sequences = new System.Collections.Generic.List<System.Collections.Generic.IList<int>>();
        var seqVec = new SizeTVectorVector();
        CopyVariableValueTo(outputVariable, seqVec);
        foreach(var seq in seqVec)
        {
            var seqList = new System.Collections.Generic.List<int>(seq.Count);
            foreach (var element in seq)
            {
                seqList.Add((int)element);
            }
            sequences.Add(seqList);
        }
        return sequences;
    }

    //
    // Copy the data of the Value object into the buffer provided by 'sequences'.
    // The 'sequences' is a list of sequences with variable length. 
    // The number of items contained in the outer list of 'sequences' is the number of sequences in the Value object.
    // Each element of the outer list represents a sequence.
    // Each sequence, represented by List<T>, contains a variable number of samples. 
    // Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the variable shape.
    // The number of samples = the count of elements in List<T> / the count of elements of the sample
    // The shape of the variable should match the shape of the Value object.
    //
    [System.Obsolete("CopyVariableValueTo() will be deprecated soon. Please use GetDenseData() instead.")]
    public void CopyVariableValueTo<T>(Variable outputVariable, System.Collections.Generic.List<System.Collections.Generic.List<T>> sequences)
    {
        sequences.Clear();
        if (typeof(T).Equals(typeof(float)))
        {
            if (GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new FloatVectorVector();
            CopyVariableValueToFloat(outputVariable, seqVec);

            foreach (var seq in seqVec)
            {
                var seqList = seq as System.Collections.Generic.IList<T>;
                if (seqList == null)
                    throw new System.TypeAccessException("Cannot convert to the value type.");
                sequences.Add(new System.Collections.Generic.List<T>(seqList));
            }
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            if (GetDataType() != DataType.Double)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new DoubleVectorVector();
            CopyVariableValueToDouble(outputVariable, seqVec);
            foreach (var seq in seqVec)
            {
                var seqList = seq as System.Collections.Generic.IList<T>;
                if (seqList == null)
                    throw new System.TypeAccessException("Cannot convert to the value type.");
                sequences.Add(new System.Collections.Generic.List<T>(seqList));
            }
        }
        else
        {
            throw new System.ArgumentException("The value type does not match the list type.");
        }
    }

    //
    // Copy the data of the Value object into the buffer provided by 'sequences'.
    // The 'sequences' is a list of sequences with variable length.
    // The number of items contained in the outer list of 'sequences' is the number of sequences in the Value object.
    // Each element of the outer list represents a sequence.
    // Each sequence, represented by List<int>, contains a variable number of samples.
    // Each sample is represented by an index of the OneHot vector. The size of the OneHot vector should match that defined in the variable. 
    // The number of samples = the count of elements in List<int>.
    //
    [System.Obsolete("CopyVariableValueTo() will be deprecated soon. Please use GetOneHotData() instead.")]
    public void CopyVariableValueTo(Variable outputVariable, System.Collections.Generic.List<System.Collections.Generic.List<int>> sequences)
    {
        var seqVec = new SizeTVectorVector();
        CopyVariableValueTo(outputVariable, seqVec);

        sequences.Clear();
        foreach(var seq in seqVec)
        {
            var seqList = new System.Collections.Generic.List<int>(seq.Count);
            foreach (var element in seq)
            {
                seqList.Add((int)element);
            }
            sequences.Add(seqList);
        }
        return;
    }

    private static FloatVector AsFloatVector<T>(System.Collections.Generic.IEnumerable<T> input)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            var inputVector = new FloatVector();
            System.Collections.Generic.IEnumerable<float> inputInType = input as System.Collections.Generic.IEnumerable<float>;
            if (inputInType == null)
                throw new System.ArgumentNullException("The parameter cannot be cast as IEnumerable<float>.");
            foreach (var element in inputInType)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    private static DoubleVector AsDoubleVector<T>(System.Collections.Generic.IEnumerable<T> input)
    {
        if (typeof(T).Equals(typeof(double)))
        {
            var inputVector = new DoubleVector();
            System.Collections.Generic.IEnumerable<double> inputInType = input as System.Collections.Generic.IEnumerable<double>;
            if (inputInType == null)
                throw new System.ArgumentNullException("The parameter cannot be cast as IEnumerable<double>.");
            foreach (var element in inputInType)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    private static SizeTVector AsSizeTVector(System.Collections.Generic.IEnumerable<int> input)
    {
        var inputVector = new SizeTVector();
        foreach (var element in input)
        {
            inputVector.Add((uint)element);
        }
        return inputVector;
    }

    private static BoolVector AsBoolVector(System.Collections.Generic.IEnumerable<bool> input)
    {
        var inputVector = new BoolVector();
        foreach (var element in input)
        {
            inputVector.Add(element);
        }
        return inputVector;
    }
%}

%include "CNTKValueExtend.i"

//
// NDArryView
//
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

%typemap(cscode) CNTK::NDArrayView %{
    public NDArrayView(NDShape viewShape, float[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
    {
    }

    public NDArrayView(NDShape viewShape, double[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
    {
    }

    public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, float[] nonZeroValues, DeviceDescriptor device, bool readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.Length, device, readOnly)
    {
        if (rowIndices.Length != nonZeroValues.Length)
        {
            throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (viewShape[viewShape.Rank-1] + 1 != colStarts.Length)
        {
            throw new System.ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");
        }
    }

    public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, double[] nonZeroValues, DeviceDescriptor device, bool readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.Length, device, readOnly)
    {
        if (rowIndices.Length != nonZeroValues.Length)
        {
            throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (viewShape[viewShape.Rank-1] + 1 != colStarts.Length)
        {
            throw new System.ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");
        }
    }

    public DeviceDescriptor Device
    {
        get
        {
            return GetDevice();
        }
    }

    public DataType DataType
    {
        get
        {
            return GetDataType();
        }
    }

    public NDShape Shape
    {
        get
        {
            return GetShape();
        }
    }

    public StorageFormat StorageFormat
    {
        get
        {
            return GetStorageFormat();
        }
    }

    public bool IsSparse
    {
        get
        {
            return _IsSparse();
        }
    }

    public bool IsReadOnly
    {
        get
        {
            return _IsReadOnly();
        }
    }

    public NDArrayView SliceView(System.Collections.Generic.IEnumerable<int> startOffset, System.Collections.Generic.IEnumerable<int> extent, bool readOnly = false)
    {
        var startOffsetVector = AsSizeTVector(startOffset);

        var extentVector = AsSizeTVector(extent);

        return _SliceView(startOffsetVector, extentVector, readOnly);
    }

    private static SizeTVector AsSizeTVector(System.Collections.Generic.IEnumerable<int> input)
    {
        var inputVector = new SizeTVector();
        foreach (var element in input)
        {
            if (element < 0)
            {
                throw new System.ArgumentException("The argument cannot contain a negative value");
            }
            inputVector.Add((uint)element);
        }
        return inputVector;
    }
%}



%ignore CNTK::Function::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(), const Internal::UDFDeserializerPtr& deserializer);
%ignore CNTK::Function::Load(const char* buffer, size_t length, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(), const Internal::UDFDeserializerPtr& deserializer = nullptr);
// Ignore exposing istream to C# for now. Todo: find a good solution to map C# System.IO.Stream to std::istream.
%ignore CNTK::Function::Load(std::istream& inputStream, const DeviceDescriptor& computeDevice= DeviceDescriptor::UseDefaultDevice(), const Internal::UDFDeserializerPtr& deserializer = nullptr);

%extend CNTK::Function {
    static FunctionPtr Load(const std::wstring& filepath, 
                            const CNTK::DeviceDescriptor& computeDevice = CNTK::DeviceDescriptor::UseDefaultDevice()) 
    {
        return CNTK::Function::Load(filepath, computeDevice, nullptr);
    }

    static FunctionPtr Load(const char* modelBuffer, size_t length,
                            const CNTK::DeviceDescriptor& computeDevice = CNTK::DeviceDescriptor::UseDefaultDevice()) 
    {
        return CNTK::Function::Load(modelBuffer, length, computeDevice, nullptr);
    }
}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

