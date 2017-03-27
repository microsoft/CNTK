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

%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(NDArrayViewVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(BoolVector) std::vector<bool>;
%template(IntVector) std::vector<int>; 
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

// Ignore things in CNTKLibrary.h that are not exposed for C# Eval.
%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);

%ignore_function CNTK::Internal::GenerateUid;

%ignore_function CNTK::PlaceholderVariable;
%ignore_function CNTK::InputVariable;
%ignore_function CNTK::OutputVariable;
%ignore_function CNTK::Internal::AddProgressWriters;

%ignore_class CNTK::Variable::CompositeFunction;
%ignore_class CNTK::Variable::Trainer;
%ignore_class CNTK::Varaiable::PrimitiveFunction;

%ignore_class CNTK::IDictionarySerializable;
%ignore_class CNTK::DictionaryValue;
%ignore_class CNTK::Dictionary;
%ignore_class CNTK::ParameterInitializer;

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
%ignore_class CNTK::PoolingType;

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
%ignore_function CNTK::ClassificationError;
%ignore_function CNTK::PastValue;
%ignore_function CNTK::FutureValue;
%ignore_function CNTK::ReduceSum;
%ignore_function CNTK::ReduceLogSum;
%ignore_function CNTK::ReduceMean;
%ignore_function CNTK::ReduceMax;
%ignore_function CNTK::ReduceMin;
%ignore_function CNTK::PerDimMeanVarianceNormalize;
%ignore_function CNTK::Convolution;
%ignore_function CNTK::ROIPooling;
%ignore_function CNTK::Pooling;
%ignore_function CNTK::Unpooling;
%ignore_function CNTK::LambdaRank;
%ignore_function CNTK::NDCGAt1;
%ignore_function CNTK::BatchNormalization;
%ignore_function CNTK::OptimizedRNNStack;
%ignore_function CNTK::Clip;
%ignore_function CNTK::ElementSelect;
%ignore_function CNTK::Splice;
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
%ignore_class CNTK::LearningRateSchedule;
%ignore_class CNTK::LearningRatePerSampleSchedule;
%ignore_class CNTK::LearningRatePerMinibatchSchedule;
%ignore_class CNTK::MomentumAsTimeConstantSchedule;
%ignore_struct CNTK::AdditionalLearningOptions;
%ignore_class CNTK::Learner;

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
%ignore_struct CNTK::StreamInformation;
%ignore_struct std::hash<::CNTK::StreamInformation>;

%ignore_struct CNTK::MinibatchData;
%ignore_struct CNTK::MinibatchSource;
%ignore_struct CNTK::MinibatchInfo;

%ignore_function CNTK::CreateCompositeMinibatchSource;
%ignore_struct CNTK::StreamConfiguration;
%ignore_function CNTK::TextFormatMinibatchSource;
%ignore_function CNTK::ComputeInputPerDimMeansAndInvStdDevs;
%ignore_struct CNTK::DistributedWorkerDescriptor;
%ignore_class CNTK::DistributedCommunicator;
%ignore_class CNTK::QuantizedDistributedCommunicator;
%ignore_function CNTK::MPICommunicator;
%ignore_function CNTK::QuantizedMPICommunicator;

%ignore_class CNTK::TrainingSession;
%ignore_function CNTK::CreateBasicTrainingSession;
%ignore_function CNTK::CreateTrainingSession;
%ignore_function CNTK::CreateDataParallelDistributedTrainer;
%ignore_function CNTK::CreateQuantizedDataParallelDistributedTrainer;

%ignore_class CNTK::ProgressWriter;

%ignore_struct std::hash<::CNTK::DistributedWorkerDescriptor>;

// Todo: add correct typemap as they might be useful for C# in future.
%ignore_function CNTK::NDMask::DataBuffer;

// Ignore things in CNTKLibraryInternals.h that are not exposed for C# Eval.
%ignore_class CNTK::Internal::PrimitiveFunction;
%ignore_class CNTK::Internal::CompositeFunction;
%ignore_function CNTK::Internal::MaxNumCPUThreadsSet;
%ignore_class CNTK::PrimitiveOpType;
%ignore_function CNTK::Internal::IsWithin;
%ignore_function CNTK::Internal::PackedIndex;
%ignore_function CNTK::Internal::GatherPacked;
%ignore_function CNTK::Internal::ScatterPacked;
%ignore_function CNTK::Internal::ZeroesWithDynamicAxesLike;
%ignore_function CNTK::Internal::Where;
%ignore_function CNTK::Internal::Gather;
%ignore_function CNTK::Internal::Scatter;
%ignore_function CNTK::Internal::Slice;
%ignore_function CNTK::Internal::ReduceElements;
%ignore_function CNTK::Internal::CosineDistanceWithNegativeSamples;

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
%ignore_function CNTK::Internal::SetFixedRandomSeed;
%ignore_function CNTK::Internal::EnableForwardValuesSharing;
%ignore_function CNTK::Internal::DisableForwardValuesSharing;
%ignore CNTK::Internal::DefaultProfilerBufferSize;
%ignore_function CNTK::Internal::StartProfiler;
%ignore_function CNTK::Internal::StopProfiler;
%ignore_function CNTK::Internal::EnableProfiler;
%ignore_function CNTK::Internal::DisableProfiler;
%ignore_function CNTK::Internal::AreEquivalent;
%ignore_function CNTK::Internal::AreEqual;
%ignore_function CNTK::PrintBuiltInfo;

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
%rename (AreEqualDeviceDescriptor) CNTK::operator==(const DeviceDescriptor& left, const DeviceDescriptor& right);

%typemap(cscode) CNTK::DeviceDescriptor %{

    // Remove this for now, will be added back after we find a good solution here:
    // This is a reference to prevent premature garbage collection 
    // and resulting in dangling access to device.
    // private static DeviceDescriptorVector deviceVector;
    // private static System.Collections.Generic.List<DeviceDescriptor> deviceList;
    // private static System.Object deviceVectorInitLock = new System.Object();

    public uint Id
    {
        get { return GetId(); }
    }

    public DeviceKind Type
    {
        get { return GetDeviceType(); }
    }

    public static DeviceDescriptor CPUDevice
    {
        get { return GetCPUDevice(); }
    }

    //public static System.Collections.Generic.List<DeviceDescriptor> AllDevices()
    //{
    //    lock (deviceVectorInitLock)
    //    {
    //        // TODO: support devices added/removed after creation. 
    //        if (deviceVector == null)
    //        {
    //            deviceVector = GetAllDevices();
    //            deviceList = new System.Collections.Generic.List<DeviceDescriptor>(deviceVector.Count);
    //            foreach (var d in deviceVector)
    //            {
    //                deviceList.Add(d);
    //            }
    //        }
    //    }
    //    return deviceList;
    //}

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
%}

%rename (GetName) CNTK::Axis::Name;
%rename (IsOrderedAxis) CNTK::Axis::IsOrdered;
%rename (AreEqualAxis) CNTK::operator==(const Axis& first, const Axis& second);

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

// Ignore exposing istream to C# for now. Todo: find a good solution to map C# System.IO.Stream to std::istream.
%ignore CNTK::Function::LoadModel(std::istream& inputStream, const DeviceDescriptor& computeDevice= DeviceDescriptor::UseDefaultDevice());

%rename (GetName) CNTK::Function::Name;
%rename (GetUid) CNTK::Function::Uid;
%rename (GetRootFunction) CNTK::Function::RootFunction;
%rename (GetInputs) CNTK::Function::Inputs;
%rename (GetOutput) CNTK::Function::Output;
%rename (GetOutputs) CNTK::Function::Outputs;
%rename (GetArguments) CNTK::Function::Arguments;
%rename (GetOpName) CNTK::Function::OpName;
%rename (_IsComposite) CNTK::Function::IsComposite;
%rename (_IsPrimitive) CNTK::Function::IsPrimitive;
%rename (_IsBlock) CNTK::Function::IsBlock;

// Customize type mapping for modelBuffer, used by LoadModel
%apply char* INPUT { char* modelBuffer }
%typemap(ctype) (char* modelBuffer) "char*"
%typemap(imtype) (char* modelBuffer) "byte[]"
%typemap(cstype) (char* modelBuffer) "byte[]"

%typemap(cscode) CNTK::Function %{

    // This is a reference to prevent premature garbage collection 
    // and resulting in dangling access to Variable.
    private VariableVector argumentVector;
    private VariableVector outputVector;
    private System.Collections.Generic.List<Variable> argumentList;
    private System.Collections.Generic.List<Variable> outputList;
    private UnorderedMapVariableValuePtr outMap = new UnorderedMapVariableValuePtr();

    public static Function LoadModel(byte[] modelBuffer, DeviceDescriptor computeDevice)
    {
        return LoadModel(modelBuffer, (uint)modelBuffer.Length, computeDevice);
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
        get 
        {
            return GetRootFunction();
        }
    }

    public System.Collections.Generic.List<Variable> Outputs
    {
        get 
        {
            // Assuming that outputs of Function can not be changed after creation.
            if (outputVector == null)
            {
                outputVector = GetOutputs();
                outputList = new System.Collections.Generic.List<Variable>(outputVector.Count);
                foreach (var v in outputVector)
                {
                    outputList.Add(v);
                }
            }
            return outputList;
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

    public System.Collections.Generic.List<Variable> Arguments
    {
        get
        {
            // Assuming that arguments of Function can not be changed after creation.
            if (argumentVector == null)
            {
                argumentVector = GetArguments();
                argumentList = new System.Collections.Generic.List<Variable>(argumentVector.Count);
                foreach (var v in argumentVector)
                {
                    argumentList.Add(v);
                }
            }
            return argumentList;
        }
    }

    // Todo: do we have a better place to put this function?
    public static Function Combine(System.Collections.Generic.IEnumerable<Variable> outputVariable)
    {
        var varVect = new VariableVector();
        foreach (var v in outputVariable)
        {
            varVect.Add(v);
        }
        return CNTKLib.Combine(varVect);
    }

    public void Evaluate(System.Collections.Generic.Dictionary<Variable, Value> arguments, System.Collections.Generic.Dictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
    {
        // Evaluate the rootFunction.
        var argMap = new UnorderedMapVariableValuePtr();
        foreach (var p in arguments)
        {
            argMap.Add(p.Key, p.Value);
        }

        outMap.Clear();
        foreach (var p in outputs)
        {
            outMap.Add(p.Key, p.Value);
        }

        Evaluate(argMap, outMap, computeDevice);

        foreach (var p in outMap)
        {
            outputs[p.Key] = p.Value;
        }
    }
%}

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

    public System.Collections.Generic.List<Axis> DynamicAxes
    {
        get {
            var axes = new System.Collections.Generic.List<Axis>();
            foreach (var axis in GetDynamicAxes())
            {
                axes.Add(axis);
            }
            return axes;
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

%typemap(cscode) CNTK::NDShape %{

    public uint Rank
    {
        get { return GetRank(); }
    }

    public System.Collections.Generic.List<uint> Dimensions
    {
        get 
        { 
            var ret = new System.Collections.Generic.List<uint>((int)GetRank());
            foreach (var dim in GetDimensions())
            {
                ret.Add((uint)dim);
            }
            return ret;
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

    public uint TotalSize
    {
        get { return GetTotalSize(); }
    }

    public uint this[int key]
    {
        get { return GetDimensionSize((uint)key); }
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

    public uint MaskedCount
    {
        get
        {
            return _MaskedCount();
        }
    }

    // Create Value object from dense input: batch, sequence or batch of sequences.
    public static Value CreateBatch<T>(NDShape sampleShape, System.Collections.Generic.List<T> batch, DeviceDescriptor device, bool readOnly = false)
    {
        var shapeSize = sampleShape.TotalSize;

        if (batch.Count % shapeSize != 0)
            throw new System.ArgumentException("The number of elements in the batch must be a multiple of the size of the shape");
        var count = batch.Count / shapeSize;
        var input = new System.Collections.Generic.List<System.Collections.Generic.List<T>>((int)count);
        for (int i = 0; i < count; i++)
        {
            var seq = new System.Collections.Generic.List<T>();
            seq.AddRange(batch.GetRange((int)(i * shapeSize), (int)shapeSize));
            input.Add(seq);
        }
        // Pass the empty sequenceStartFlags means all sequences have the start flag with true.
        return Create<T>(sampleShape, input, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

     public static Value CreateSequence<T>(NDShape sampleShape,
                                          System.Collections.Generic.List<T> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(sampleShape, sequence, true, device, readOnly);
    }

    public static Value CreateSequence<T>(NDShape sampleShape,
                                          System.Collections.Generic.List<T> sequence,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        var input = new System.Collections.Generic.List<System.Collections.Generic.List<T>>(1) {sequence};
        return Create(sampleShape, input, new System.Collections.Generic.List<bool>(1) {sequenceStartFlag}, device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                  System.Collections.Generic.List<System.Collections.Generic.List<T>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(sampleShape, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                  System.Collections.Generic.List<System.Collections.Generic.List<T>> batchOfSequences,
                                                  System.Collections.Generic.List<bool> sequenceStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    private static Value Create<T>(NDShape sampleShape,
                                  System.Collections.Generic.List<System.Collections.Generic.List<T>> sequences,
                                  System.Collections.Generic.List<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = new BoolVector(sequenceStartFlags);
        if (typeof(T).Equals(typeof(float)))
        {
            var inputSeqVector = new FloatVectorVector();
            var floatVectorRefList = new System.Collections.Generic.List<FloatVector>();
            foreach (var seq in sequences)
            {
                var seqFloatVector = new FloatVector(seq);
                floatVectorRefList.Add(seqFloatVector);
                inputSeqVector.Add(seqFloatVector);
            }
            return Value.CreateDenseFloat(sampleShape, inputSeqVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputSeqVector = new DoubleVectorVector();
            var doubleVectorRefList = new System.Collections.Generic.List<DoubleVector>();
            foreach (var seq in sequences)
            {
                var seqDoubleVector = new DoubleVector(seq);
                doubleVectorRefList.Add(seqDoubleVector);
                inputSeqVector.Add(seqDoubleVector);
            }
            return Value.CreateDenseDouble(sampleShape, inputSeqVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input, for N-dimenstional tensor. Only Create() method for now.
    private static Value Create<T>(NDShape sampleShape,
                                  System.Collections.Generic.List<System.Collections.Generic.List<uint>> sequences,
                                  System.Collections.Generic.List<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = new BoolVector(sequenceStartFlags);
        var inputSeqVector = new SizeTVectorVector();
        var sizeTVectorRefList = new System.Collections.Generic.List<SizeTVector>();
        foreach (var seq in sequences)
        {
            var s = new SizeTVector(seq);
            sizeTVectorRefList.Add(s);
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
    public static Value CreateBatch<T>(uint dimension, System.Collections.Generic.List<uint> batch, DeviceDescriptor device, bool readOnly = false)
    {
        // Is CreateBatch for OneHot really useful? 
        var input = new System.Collections.Generic.List<System.Collections.Generic.List<uint>>();
        batch.ForEach(element => input.Add(new System.Collections.Generic.List<uint>(1) {element}));

        return Create<T>(dimension, input, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value CreateSequence<T>(uint dimension,
                                          System.Collections.Generic.List<uint> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(dimension, sequence, true, device, readOnly);
    }

    public static Value CreateSequence<T>(uint dimension,
                                          System.Collections.Generic.List<uint> sequence,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        var input = new System.Collections.Generic.List<System.Collections.Generic.List<uint>>(1) {sequence};
        return Create<T>(dimension, input, new System.Collections.Generic.List<bool>(1) {sequenceStartFlag}, device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(uint dimension,
                                                  System.Collections.Generic.List<System.Collections.Generic.List<uint>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(uint dimension, 
                                                  System.Collections.Generic.List<System.Collections.Generic.List<uint>> batchOfSequences,
                                                  System.Collections.Generic.List<bool> sequenceStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    private static Value Create<T>(uint dimension,
                                  System.Collections.Generic.List<System.Collections.Generic.List<uint>> sequences,
                                  System.Collections.Generic.List<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = new BoolVector(sequenceStartFlags);
        var inputSeqVector = new SizeTVectorVector();
        var sizeTVectorRefList = new System.Collections.Generic.List<SizeTVector>();
        foreach (var seq in sequences)
        {
            var s = new SizeTVector(seq);
            sizeTVectorRefList.Add(s);
            inputSeqVector.Add(s);
        }
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateOneHotFloat(dimension, inputSeqVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateOneHotDouble(dimension, inputSeqVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from sparse input, for N-dimensional tensor. Only CreateSequence() for now.
    public static Value CreateSequence<T>(NDShape sampleShape, uint sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues, uint numNonZeroValues,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateSequenceFloat(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues as float[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateSequenceDouble(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues as double[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateSequence<T>(NDShape sampleShape, uint sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues, uint numNonZeroValues,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return Value.CreateSequence<T>(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, true, device, readOnly);
    }

    // Create Value object from sparse input, for 1D tensor. Only CreateSequence() for now.
    public static Value CreateSequence<T>(uint dimension, uint sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues, uint numNonZeroValues,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            return Value.CreateSequenceFloat(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues as float[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value.CreateSequenceDouble(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues as double[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value CreateSequence<T>(uint dimension, uint sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues, uint numNonZeroValues,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return Value.CreateSequence<T>(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, true, device, readOnly);
    }

    // Create value object from NDArrayView
    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.List<NDArrayView> sequences,
                               DeviceDescriptor device,
                               bool readOnly = false)
    {
        return Create(sampleShape, sequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.List<NDArrayView> sequences,
                               System.Collections.Generic.List<bool> sequenceStartFlags,
                               DeviceDescriptor device,
                               bool readOnly = false)
    {
        var seqVector = new NDArrayViewVector(sequences);
        var startVector = new BoolVector(sequenceStartFlags);
        return Create(sampleShape, seqVector, startVector, device, false);
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
    public void CopyVariableValueTo<T>(Variable outputVariable, System.Collections.Generic.List<System.Collections.Generic.List<T>> sequences)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            if (GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new FloatVectorVector();
            CopyVariableValueToFloat(outputVariable, seqVec);
            sequences.Clear();
            foreach (var seq in seqVec)
            {
                var seqList = seq as System.Collections.Generic.IEnumerable<T>;
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
            sequences.Clear();
            foreach (var seq in seqVec)
            {
                var seqList = seq as System.Collections.Generic.IEnumerable<T>;
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
    // Each sequence, represented by List<uint>, contains a variable number of samples. 
    // Each sample is represented by an index of the OneHot vector. The size of the OneHot vector should match that defined in the variable. 
    // The number of samples = the count of elements in List<uint>.
    //
    public void CopyVariableValueTo(Variable outputVariable, System.Collections.Generic.List<System.Collections.Generic.List<uint>> sequences)
    {
        if (outputVariable.Shape[0] != outputVariable.Shape.TotalSize)
        {
            throw new System.ArgumentException("The sample variable's leading axis dimensionality must equal to the total size of the shape for sparse data");
        }

        var seqVec = new SizeTVectorVector();
        CopyVariableValueTo(outputVariable, seqVec);

        sequences.Clear();
        foreach(var seq in seqVec)
        {
            sequences.Add(new System.Collections.Generic.List<uint>(seq));
        }
        return;
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
        return new CNTK::NDArrayView(CNTK::DataType::Float, viewShape, dataBuffer, numBufferElements * sizeof(float), device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, double *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Double, viewShape, dataBuffer, numBufferElements * sizeof(double), device, readOnly);
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

// 
// NDShape
//
%extend CNTK::NDShape {
    size_t GetDimensionSize(size_t axisId)
    {
        return (*self)[axisId];
    }
}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

