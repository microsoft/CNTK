%module(directors="1") CNTKLib
//%feature("autodoc", "1");
%feature("director") Learner;

%include <stl.i>
%include <std_common.i>
%include <std_wstring.i>
%include <std_vector.i>
%include <std_map.i>
%include <std_pair.i>
%include <std_shared_ptr.i>
%include <windows.i>
%include <attribute.i>
%include <arrays_csharp.i>
#include <exception.i>

%apply void *VOID_INT_PTR { void * }

// map the pointer to array
%apply int INPUT[]  { int *dataBuffer }
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }

%apply int OUTPUT[] { int *dataBuffer }
%apply float OUTPUT[]  { float *dataBuffer }
%apply double OUTPUT[]  { double *dataBuffer }

INPUT_TYPEMAP(uint64_t, uint64_t, ulong);

%include "std_unordered_map.i"
%include "std_unordered_set.i"

%{
    #include "CNTKLibrary.h"	
    #pragma warning(disable : 4100)
%}

%typemap(csclassmodifiers) SWIGTYPE "public partial class"

%template(BoolVector) std::vector<bool>;
%template(SizeTVector) std::vector<size_t>;
%template(FloatVector) std::vector<float>;
%template(DoubleVector) std::vector<double>;
%template(BoolVectorVector) std::vector<std::vector<bool>>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;

%template(SizeTDoublePair) std::pair<size_t, double>;
%template(SizeTSizeTPair) std::pair<size_t, size_t>;
%template(SizeTIntPair) std::pair<size_t, int>;
%template(SizeTDoublePairVector) std::vector<std::pair<size_t, double>>;
%template(SizeTSizeTPairVector) std::vector<std::pair<size_t, size_t>>;

%shared_ptr(std::vector<bool>);
%shared_ptr(std::vector<size_t>);
%shared_ptr(std::vector<float>);
%shared_ptr(std::vector<double>);
%shared_ptr(std::pair<size_t, int>);

%shared_ptr(CNTK::BackPropState);
%shared_ptr(CNTK::CompositeFunction);
%shared_ptr(CNTK::NDShape);
%shared_ptr(CNTK::NDArrayView);
%shared_ptr(CNTK::NDMask);
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::Function);
%shared_ptr(CNTK::Learner);
%shared_ptr(CNTK::Dictionary);
%shared_ptr(CNTK::DictionaryValue);
%shared_ptr(CNTK::MinibatchSource);
%shared_ptr(CNTK::DistributedLearner);
%shared_ptr(CNTK::DistributedCommunicator);
%shared_ptr(CNTK::QuantizedDistributedCommunicator);
%shared_ptr(CNTK::VariableFields);
%shared_ptr(CNTK::TrainingSession);
%shared_ptr(CNTK::Trainer);

%shared_ptr(std::shared_ptr<CNTK::Function>);

%template(AxisVector) std::vector<CNTK::Axis>;
%template(ConstantVector) std::vector<CNTK::Constant>;
%template(ParameterVector) std::vector<CNTK::Parameter>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(DictionaryVector) std::vector<std::shared_ptr<CNTK::Dictionary>>;
%template(DictionaryValueVector) std::vector<::CNTK::DictionaryValue>;
%template(DictionaryValueVector) std::vector<CNTK::DictionaryValue>;
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;
%template(VariableVariablePair) std::pair<CNTK::Variable, CNTK::Variable>;
%template(VariableVariablePairVector) std::vector<std::pair<CNTK::Variable, CNTK::Variable>>;
%template(FunctionVector) std::vector<std::shared_ptr<CNTK::Function>>;
%template(ValueVector) std::vector<std::shared_ptr<CNTK::Value>>;
%template(LearnerVector) std::vector<std::shared_ptr<CNTK::Learner>>;
%template(DistributedLearnerVector) std::vector<std::shared_ptr<CNTK::DistributedLearner>>;
%template(NDArrayViewVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(StreamConfigurationVector) std::vector<CNTK::StreamConfiguration>;

%template(NDArrayViewNDArrayViewPair) std::pair<std::shared_ptr<CNTK::NDArrayView>, std::shared_ptr<CNTK::NDArrayView>>;
%template(NDArrayViewBoolVectorPair) std::pair<std::vector<std::shared_ptr<CNTK::NDArrayView>>, std::vector<bool>>;

%template(UnorderedMapStringDictionaryValue) std::unordered_map<std::wstring, CNTK::DictionaryValue>;
%template(UnorderedMapDictionaryDictionary) std::unordered_map<std::shared_ptr<CNTK::Dictionary>, CNTK::Dictionary>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;
%template(UnorderedMapVariableValue) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableMinibatchData) std::unordered_map<CNTK::Variable, CNTK::MinibatchData>;
%template(UnorderedMapVariableStreamInformation) std::unordered_map<CNTK::Variable, CNTK::StreamInformation>;
%template(UnorderedMapFunctionSizeT) std::unordered_map<std::shared_ptr<CNTK::Function>, size_t>;
%template(UnorderedMapFunctionFunction) std::unordered_map<std::shared_ptr<CNTK::Function>, std::shared_ptr<CNTK::Function>>;
%template(UnorderedMapParameterNDArrayView) std::unordered_map<CNTK::Parameter, std::shared_ptr<CNTK::NDArrayView>>;

%template(UnorderedMapStreamInformationMinibatchData) std::unordered_map<CNTK::StreamInformation, CNTK::MinibatchData>;
%template(UnorderedMapStreamInformationNDArrayViewNDArrayViewPair) std::unordered_map<CNTK::StreamInformation, std::pair<std::shared_ptr<CNTK::NDArrayView>, std::shared_ptr<CNTK::NDArrayView>>>;

%template(UnorderedSetString) std::unordered_set<std::wstring>;
%template(UnorderedSetVariable) std::unordered_set<CNTK::Variable>;
%template(UnorderedSetParameter) std::unordered_set<CNTK::Parameter>;
%template(UnorderedSetFunction) std::unordered_set<std::shared_ptr<CNTK::Function>>;
%template(UnorderedSetStreamInformation) std::unordered_set<CNTK::StreamInformation>;
%template(UnorderedSetDistributedWorkerDescriptor) std::unordered_set<CNTK::DistributedWorkerDescriptor>;

#define %ignore_function %rename("$ignore", %$isfunction, fullname=1)
#define %ignore_class %rename("$ignore", %$isclass, fullname=1)
#define %ignore_namespace %rename("$ignore", %$isnamespace, fullname=1)
#define %ignore_variable %rename("$ignore", %$isvariable, fullname=1)
// It seems that SWIG does not understand %$isstruct.
#define %ignore_struct %rename("$ignore", fullname=1)


// Ignore things in CNTKLibrary.h that are not exposed for C# Eval.
%ignore CNTK::NDShape::NDShape(const std::initializer_list<size_t>& dimensions);

%ignore_class CNTK::IDictionarySerializable;

%ignore_function CNTK::Dictionary::begin;
%ignore_function CNTK::Dictionary::cbegin;
%ignore_function CNTK::Dictionary::end;
%ignore_function CNTK::Dictionary::cend;

%rename (GetFunction) CNTK::Function::Function;
%rename (GetFunction) CNTK::BackPropState::Function;

%ignore_function CNTK::NDArrayView::operator=;
%ignore_function CNTK::operator-;

// Todo: add correct typemap as they might be useful for C# in future.
//MaskKind*
%typemap(csbase) CNTK::MaskKind "byte"
%shared_ptr(CNTK::MaskKind);
//%template(MaskKindPtr) CNTK::MaskKind*;
%ignore_function CNTK::NDMask::DataBuffer;


%rename(Sequence_IsFirst) CNTK::Sequence::IsFirst;
%rename(Sequence_IsLast) CNTK::Sequence::IsLast;
%rename(Sequence_Slice) CNTK::Sequence::Slice;
%rename(Sequence_ReduceSum) CNTK::Sequence::ReduceSum;
%rename(Sequence_First) CNTK::Sequence::First;
%rename(Sequence_Last) CNTK::Sequence::Last;
%rename(Sequence_Where) CNTK::Sequence::Where;
%rename(Sequence_Gather) CNTK::Sequence::Gather;
%rename(Sequence_Scatter) CNTK::Sequence::Scatter;
%rename(Sequence_BroadcastAs) CNTK::Sequence::BroadcastAs;

%ignore_function CNTK::Internal::MaxNumCPUThreadsSet;//unresolved external
%ignore_function CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled;//unresolved external
%ignore_function CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed;//
%ignore_function CNTK::Internal::IsRenamingFunctionsAllowed;
%ignore_function CNTK::Internal::GetComputationNetworkTraceLevel;

%rename(Internal_GenerateUid) CNTK::Internal::GenerateUid;
%rename(Internal_IsWithin) CNTK::Internal::IsWithin;
%rename(Internal_PackedIndex) CNTK::Internal::PackedIndex;
%rename(Internal_GatherPacked) CNTK::Internal::GatherPacked;
%rename(Internal_ScatterPacked) CNTK::Internal::ScatterPacked;
%rename(Internal_ZeroesWithDynamicAxesLike) CNTK::Internal::ZeroesWithDynamicAxesLike;
%rename(Internal_Where) CNTK::Internal::Where;
%rename(Internal_Gather) CNTK::Internal::Gather;
%rename(Internal_Scatter) CNTK::Internal::Scatter;
%rename(Internal_ReduceElements) CNTK::Internal::ReduceElements;
%rename(Internal_Slice) CNTK::Internal::Slice;

%rename(Internal_EnableReversingTensorShapesInErrorMessages) CNTK::Internal::EnableReversingTensorShapesInErrorMessages;
%rename(Internal_AlwaysAllowSettingDefaultDevice) CNTK::Internal::AlwaysAllowSettingDefaultDevice;
%rename(Internal_AllowRenamingFunctions) CNTK::Internal::AllowRenamingFunctions;
%rename(Internal_SetAutomaticUnpackingOfPackedValues) CNTK::Internal::SetAutomaticUnpackingOfPackedValues;
%rename(Internal_IsAutomaticUnpackingOfPackedValuesDisabled) CNTK::Internal::IsAutomaticUnpackingOfPackedValuesDisabled;
%rename(Internal_SetComputationNetworkTraceLevel) CNTK::Internal::SetComputationNetworkTraceLevel;

%rename(Internal_SetGPUMemoryAllocationTraceLevel) CNTK::Internal::SetGPUMemoryAllocationTraceLevel;
%rename(Internal_ForceSynchronousCUDAKernelExecutions) CNTK::Internal::ForceSynchronousCUDAKernelExecutions;
%rename(Internal_ForceDeterministicAlgorithms) CNTK::Internal::ForceDeterministicAlgorithms;
%rename(Internal_SetFixedRandomSeed) CNTK::Internal::SetFixedRandomSeed;
%rename(Internal_EnableForwardValuesSharing) CNTK::Internal::EnableForwardValuesSharing;
%rename(Internal_DisableForwardValuesSharing) CNTK::Internal::DisableForwardValuesSharing;
%rename(Internal_EnableHyperMemoryCompress) CNTK::Internal::EnableHyperMemoryCompress;
%rename(Internal_DisableHyperMemoryCompress) CNTK::Internal::DisableHyperMemoryCompress;
%rename(Internal_StartProfiler) CNTK::Internal::StartProfiler;
%rename(Internal_StopProfiler) CNTK::Internal::StopProfiler;
%rename(Internal_EnableProfiler) CNTK::Internal::EnableProfiler;
%rename(Internal_DisableProfiler) CNTK::Internal::DisableProfiler;
%rename(Internal_AreEquivalent) CNTK::Internal::AreEquivalent;
%rename(Internal_AreEqual) CNTK::Internal::AreEqual;

%ignore CNTK::Internal::DefaultProfilerBufferSize;
//%rename(Internal_IsReversingTensorShapesInErrorMessagesEnabled) CNTK::Internal::IsReversingTensorShapesInErrorMessagesEnabled;
//%rename(Internal_IsSettingDefaultDeviceAlwaysAllowed) CNTK::Internal::IsSettingDefaultDeviceAlwaysAllowed;
//%rename(Internal_IsRenamingFunctionsAllowed) CNTK::Internal::IsRenamingFunctionsAllowed;
//%rename(Internal_GetComputationNetworkTraceLevel) CNTK::Internal::GetComputationNetworkTraceLevel;

// Exception handling
%include "CNTK_ExceptionHandling.i"

%rename (GetAllDevices) CNTK::DeviceDescriptor::AllDevices;
%rename (GetCPUDevice) CNTK::DeviceDescriptor::CPUDevice;
%rename (GetDeviceType) CNTK::DeviceDescriptor::Type;
%rename (GetId) CNTK::DeviceDescriptor::Id;
%rename (AreEqualDeviceDescriptor) CNTK::operator==(const DeviceDescriptor& left, const DeviceDescriptor& right);

%rename (DoFinalize) CNTK::DistributedCommunicator::Finalize;


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

%typemap(cscode) CNTK::Function %{

    // This is a reference to prevent premature garbage collection 
    // and resulting in dangling access to Variable.
    private VariableVector argumentVector;
    private VariableVector outputVector;
    private System.Collections.Generic.List<Variable> argumentList;
    private System.Collections.Generic.List<Variable> outputList;
    private UnorderedMapVariableValue outMap = new UnorderedMapVariableValue();

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
        var argMap = new UnorderedMapVariableValue();
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
    public static Value CreateBatch<T>(NDShape shape, System.Collections.Generic.List<T> batch, DeviceDescriptor device, bool readOnly = false)
    {
        var shapeSize = shape.TotalSize;

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
        // Pass the empty seqStartFlags means all sequences have the start flag with true.
        return Create<T>(shape, input, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

     public static Value CreateSequence<T>(NDShape shape,
                                          System.Collections.Generic.List<T> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(shape, sequence, true, device, readOnly);
    }

    public static Value CreateSequence<T>(NDShape shape,
                                          System.Collections.Generic.List<T> sequence,
                                          bool seqStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        var input = new System.Collections.Generic.List<System.Collections.Generic.List<T>>(1) {sequence};
        return Create(shape, input, new System.Collections.Generic.List<bool>(1) {seqStartFlag}, device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(NDShape shape,
                                                  System.Collections.Generic.List<System.Collections.Generic.List<T>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(shape, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    public static Value CreateBatchOfSequences<T>(NDShape shape,
                                                  System.Collections.Generic.List<System.Collections.Generic.List<T>> batchOfSequences,
                                                  System.Collections.Generic.List<bool> seqStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(shape, batchOfSequences, seqStartFlags, device, readOnly);
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

    // Create Value object from OneHotVector input: batch, sequence or batch of sequences
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
                                          bool seqStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        var input = new System.Collections.Generic.List<System.Collections.Generic.List<uint>>(1) {sequence};
        return Create<T>(dimension, input, new System.Collections.Generic.List<bool>(1) {seqStartFlag}, device, readOnly);
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
                                                  System.Collections.Generic.List<bool> seqStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, seqStartFlags, device, readOnly);
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
        //var seqVector = new NDArrayViewVector(sequences);
        //var startVector = new BoolVector(sequenceStartFlags);
        return Create(sampleShape, sequences, sequenceStartFlags, device, false);
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
    public void CopyVariableValueTo<T>(Variable sampleVariable, System.Collections.Generic.List<System.Collections.Generic.List<T>> sequences)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            if (GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new FloatVectorVector();
            CopyVariableValueToFloat(sampleVariable, seqVec);
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
            CopyVariableValueToDouble(sampleVariable, seqVec);
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
    public void CopyVariableValueTo(Variable sampleVariable, System.Collections.Generic.List<System.Collections.Generic.List<uint>> sequences)
    {
        if (sampleVariable.Shape[0] != sampleVariable.Shape.TotalSize)
        {
            throw new System.ArgumentException("The sample variable's leading axis dimensionality must equal to the total size of the shape for sparse data");
        }

        var seqVec = new SizeTVectorVector();
        CopyVariableValueTo(sampleVariable, seqVec);

        sequences.Clear();
        foreach(var seq in seqVec)
        {
            sequences.Add(new System.Collections.Generic.List<uint>(seq));
        }
        return;
    }



%}

%extend CNTK::Value {
    void CNTK::Value::CopyVariableValueToFloat(const CNTK::Variable& sampleVariable, std::vector<std::vector<float>>& sequences)
    {
        return self->CopyVariableValueTo<float>(sampleVariable, sequences);
    }

    void CNTK::Value::CopyVariableValueToDouble(const CNTK::Variable& sampleVariable, std::vector<std::vector<double>>& sequences)
    {
        return self->CopyVariableValueTo<double>(sampleVariable, sequences);
    }
}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%include "CNTKValueExtend.i"

//
// NDArryView
//
%extend CNTK::NDArrayView {
    NDArrayView(const NDShape& viewShape, float *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Float, viewShape, dataBuffer, numBufferElements * sizeof(float), device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, double *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Double, viewShape, dataBuffer, numBufferElements * sizeof(double), device, readOnly);
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

