%module(directors="1") CSharpBindings
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

// include the unordered_map.i.
%include "std_unordered_map.i"

%{
    #include "CNTKLibrary.h"
    #pragma warning(disable : 4100)
%}

%nspace CNTK;

%template(SizeTVector) std::vector<size_t>;
%template(BoolVector) std::vector<bool>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(SizeTVectorVector) std::vector<std::vector<size_t>>;
%template(FloatVectorVector) std::vector<std::vector<float>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(VariableVector) std::vector<CNTK::Variable>;
//%template() std::vector<CNTK::Parameter>;
//%template() std::vector<CNTK::Constant>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;
//%template() std::vector<CNTK::StreamConfiguration>;

//%template() std::vector<CNTK::DictionaryValue>;

%template() std::vector<std::shared_ptr<CNTK::Function>>;
%template() std::vector<std::shared_ptr<CNTK::Learner>>;
%template() std::pair<size_t, double>;
%template() std::vector<std::pair<size_t, double>>;

%shared_ptr(CNTK::BackPropState);
%shared_ptr(CNTK::Function);
%shared_ptr(CNTK::CompositeFunction);
%shared_ptr(CNTK::Value);
%shared_ptr(CNTK::NDShape);
%shared_ptr(CNTK::NDArrayView);
%shared_ptr(CNTK::NDMask);
%shared_ptr(std::vector<float>);

// SWIG does not understand ValuePtr here.
%template(UnorderedMapVariableValuePtr) std::unordered_map<CNTK::Variable, std::shared_ptr<CNTK::Value>>;
%template(UnorderedMapVariableVariable) std::unordered_map<CNTK::Variable, CNTK::Variable>;

%ignore CNTK::IDictionarySerializable;
%ignore CNTK::DictionaryValue;
%ignore CNTK::Dictionary;
%ignore CNTK::ParameterInitializer;

%ignore CNTK::Parameter;
%ignore CNTK::Constant;
%ignore CNTK::PoolingType;
%ignore CNTK::TrainingParameterSchedule;
%ignore CNTK::TrainingParameterPerUnitSchedule;
%ignore CNTK::MomentumAsTimeConstantSchedule;
%ignore CNTK::AdditionalLearningOptions;
%ignore CNTK::Learner;
%ignore CNTK::Trainer;
%ignore CNTK::StreamInformation;
%ignore CNTK::MinibatchData;
%ignore CNTK::MinibatchSource;
%ignore CNTK::StreamConfiguration;
%ignore CNTK::DistributedWorkerDescriptor;
%ignore CNTK::DistributedCommunicator;
%ignore CNTK::QuantizedDistributedCommunicator;
%ignore CNTK::MinibatchInfo;
%ignore CNTK::DistributedTrainer;

// map the point to array
%apply float INPUT[]  { float *dataBuffer }
%apply double INPUT[]  { double *dataBuffer }

%rename (GetAllDevices) CNTK::DeviceDescriptor::AllDevices;
%rename (GetBestDevice) CNTK::DeviceDescriptor::BestDevice;
%rename (GetDefaultDevice) CNTK::DeviceDescriptor::DefaultDevice;
%rename (GetCPUDevice) CNTK::DeviceDescriptor::CPUDevice;
%rename (GetDeviceType) CNTK::DeviceDescriptor::Type;
// %rename (GetId) CNTK::DeviceDescriptor::Id;

%typemap(cscode) CNTK::DeviceDescriptor %{

// Todo: why mapped to uint??
//    public int Id
//    {
//        get { return GetId(); }
//    }

    public DeviceKind Type
    {
        get { return GetDeviceType(); }
    }

    public static DeviceDescriptor CPUDevice
    {
        get { return GetCPUDevice(); }
    }

    public static DeviceDescriptor DefaultDevice
    {
        get { return GetDefaultDevice(); }
    }

    public static DeviceDescriptor BestDevice
    {
        get { return GetBestDevice(); }
    }

    public static System.Collections.Generic.List<DeviceDescriptor> AllDevices
    {
        get {
            var devices = GetAllDevices();
            var ret = new System.Collections.Generic.List<DeviceDescriptor>(devices.Count);
            foreach (var d in devices)
            {
                ret.Add(d);
            }
            return ret;
        }
    }
%}


%rename (GetOutput) CNTK::Function::Output;
%rename (GetOutputs) CNTK::Function::Outputs;
%rename (GetArguments) CNTK::Function::Arguments;

%typemap(cscode) CNTK::Function %{

    public System.Collections.Generic.List<Variable> Outputs
    {
        get 
        {
            var vars = GetOutputs();
            var ret = new System.Collections.Generic.List<Variable>(vars.Count);
            foreach (var v in vars)
            {
                ret.Add(v);
            }
            return ret;
        }
    }

    public Variable Output
    {
        get { return GetOutput(); }
    }

    public System.Collections.Generic.List<Variable> Arguments
    {
        get 
        {
            var vars = GetArguments();
            var ret = new System.Collections.Generic.List<Variable>(vars.Count);
            foreach (var v in vars)
            {
                ret.Add(v);
            }
            return ret;
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
        return CSharpBindings.Combine(varVect);        
    }
%}

%rename (GetShape) CNTK::Variable::Shape;
%rename (GetName) CNTK::Variable::Name;

%typemap(cscode) CNTK::Variable %{

    public NDShape Shape
    {
        get { return GetShape(); }
    }

    public string Name
    {
        get { return GetName(); }
    }
%}

%rename (GetDimensions) CNTK::NDShape::Dimensions;
%rename (GetRank) CNTK::NDShape::Rank;
%rename (GetTotalSize) CNTK::NDShape::TotalSize;
%rename (AreEqualShape) operator==(const NDShape& first, const NDShape& second);

%typemap(cscode) CNTK::NDShape %{
    public uint Rank
    {
        get { return GetRank(); }
    }

    public System.Collections.Generic.List<long> Dimensions
    {
        get 
        { 
            var ret = new System.Collections.Generic.List<long>((int)GetRank());
            foreach (var dim in GetDimensions())
            {
                ret.Add(dim);
            }
            return ret;
        }
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
        return CSharpBindings.AreEqualShape(this, p);
    }

    public bool Equals(NDShape p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CSharpBindings.AreEqualShape(this, p);
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
        return CSharpBindings.AreEqualShape(first, second);
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

%typemap(cscode) CNTK::Value %{

    public static Value Create<T>(NDShape shape, System.Collections.Generic.List<System.Collections.Generic.List<T>> sequences, DeviceDescriptor computeDevice)
    {
        var dim = shape.TotalSize;

        if (typeof(T).Equals(typeof(float)))
        {
            var inputSeqVector = new FloatVectorVector();
            foreach (var seq in sequences)
            {
                if (seq.Count % dim != 0)
                {
                    throw new System.ArgumentException("the number of data in sequences does not match the input dimension");
                }
                var samples = new FloatVector(seq);
                inputSeqVector.Add(samples);
            }
            return Value.CreateDenseFloat(shape, inputSeqVector, computeDevice);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputSeqVector = new DoubleVectorVector();
            foreach (var seq in sequences)
            {
                if (seq.Count % dim != 0)
                {
                    throw new System.ArgumentException("the number of data in sequences does not match the input dimension");
                }
                var samples = new DoubleVector(seq);
                inputSeqVector.Add(samples);
            }
            return Value.CreateDenseDouble(shape, inputSeqVector, computeDevice);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    public static Value Create<T>(long vacabSize, System.Collections.Generic.List<System.Collections.Generic.List<long>> oneHotIndex, DeviceDescriptor computeDevice)
    {
        throw new System.NotImplementedException("Not implemented");
    }

    // Create Value based on sparse input
    // Todo: could this be a extension to Value class??
    // Todo: use Variable instead of varName. VarName as extension method
    public static Value Create<T>(NDShape shape, System.Collections.Generic.List<System.Collections.Generic.List<T>> data, System.Collections.Generic.List<System.Collections.Generic.List<long>> indexes, System.Collections.Generic.List<System.Collections.Generic.List<long>> nnzCounts, DeviceDescriptor computeDevice)
    {
        throw new System.NotImplementedException("Not implemented");
    }
%}


%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

//
// Value
//
%extend CNTK::Value {
    static CNTK::ValuePtr CNTK::Value::CreateDenseFloat(const CNTK::NDShape& sampleShape, const std::vector<std::vector<float>>& sequences, 
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(sampleShape, sequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateDenseDouble(const CNTK::NDShape& sampleShape, const std::vector<std::vector<double>>& sequences, 
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(sampleShape, sequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotFloat(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, 
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(vocabularySize, oneHotSequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotDouble(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, 
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(vocabularySize, oneHotSequences, device, readOnly);
    }
}


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

//
// Function
//
%extend CNTK::Function {
    ///
    /// Computes and stores the values of specified variables in the 'outputs' map, using provided 'inputs' values corresponding
    /// to each leaf variable of the function of VariableKind 'Input'.
    /// The function does not return any variables that needed for backpropagation of gradients.
    ///
    void Evaluate(const std::unordered_map<Variable, ValuePtr>& arguments,
                    std::unordered_map<Variable, ValuePtr>& outputs,
                    const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice())
    {
        self->Forward(arguments, outputs, computeDevice, {});
    }

}




