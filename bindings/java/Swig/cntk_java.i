%{
#undef UNUSED
%}

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

%template(SizeTVector) std::vector<size_t>;
%template(DoubleVector) std::vector<double>;
%template(FloatVector) std::vector<float>;
%template(VariableVector) std::vector<CNTK::Variable>;
%template(AxisVector) std::vector<CNTK::Axis>;
%template(NDArrayViewPtrVector) std::vector<std::shared_ptr<CNTK::NDArrayView>>;
%template(BoolVector) std::vector<bool>;
%template(DeviceDescriptorVector) std::vector<CNTK::DeviceDescriptor>;

%include "managed_language_base.i"

%typemap(javacode) CNTK::DeviceDescriptor %{

    public java.util.ArrayList<DeviceDescriptor> getAllDevices() {
        DeviceDescriptorVector devices = GetAllDevices();
        java.util.ArrayList<DeviceDescriptor> ret = new java.util.ArrayList<DeviceDescriptor>((int)devices.size());
        for (int i = 0; i < devices.size(); ++i){
            ret.add(devices.get(i));
        }
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        DeviceDescriptor p = (DeviceDescriptor)o;
        if (p == null) return false;
        return Utils.AreEqualDeviceDescriptor(this, p);
    }

    public boolean equals(DeviceDescriptor p) {
        if (p == null) return false;
        return Utils.AreEqualDeviceDescriptor(this, p);
    }

    @Override
    public int hashCode() {
        return GetDeviceType().hashCode();
    }
%}

%typemap(javacode) CNTK::Axis %{
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        Axis p = (Axis)o;
        if (p == null) return false;
        return Utils.AreEqualAxis(this, p);
    }

    public boolean equals(Axis p) {
        if (p == null) return false;
        return Utils.AreEqualAxis(this, p);
    }

    @Override
    public int hashCode() {
        if (this.IsDynamicAxis()) {
            return GetName().hashCode();
        } else {
            return this.StaticAxisIndex();
        }
    }
%}

%typemap(javacode) CNTK::Function %{

    private VariableVector argumentVector;
    private VariableVector outputVector;
    private java.util.ArrayList<Variable> argumentList;
    private java.util.ArrayList<Variable> outputList;

    private UnorderedMapVariableValuePtr outMap = new UnorderedMapVariableValuePtr();

    public java.util.ArrayList<Variable> getOutputs() {
        if (outputVector == null) {
            outputVector = GetOutputs();
            outputList = new java.util.ArrayList<Variable>((int)outputVector.size());
            for (int i = 0; i < outputVector.size(); ++i){
                outputList.add(outputVector.get(i));
            }
        }
        return outputList;
    }

    public java.util.ArrayList<Variable> getArguments() {
        if (argumentVector == null) {
            argumentVector = GetArguments();
            argumentList = new java.util.ArrayList<Variable>((int)argumentVector.size());
            for (int i = 0; i < argumentVector.size(); ++i){
                argumentList.add(argumentVector.get(i));
            }
        }
        return argumentList;
    }

    // Todo: do we have a better place to put this function?
    public static Function Combine(java.util.ArrayList<Variable> outputVariable) {
        VariableVector varVect = new VariableVector();
        for (int i = 0; i < outputVariable.size(); ++i)
        {
            varVect.add(varVect.get(i));
        }
        return Utils.Combine(varVect);
    }

    /*public void evaluate(java.util.HashMap<Variable, Value> arguments, java.util.HashMap<Variable, Value> outputs, DeviceDescriptor computeDevice) {
        // Evaluate the rootFunction.
        UnorderedMapVariableValuePtr argMap = new UnorderedMapVariableValuePtr();

        for (Variable var : arguments.keySet()) {
            argMap.Add(var, arguments.get(var));
        }

        outMap.Clear();
        for (Variable var : outputs.keySet()) {
            outMap.Add(var, outputs.get(var));
        }

        Evaluate(argMap, outMap, computeDevice);

        for ( Variable var : outMap.keySet()) {
            outputs.put(var, outMap.get(var));
        }
    }*/
%}


%typemap(javacode) CNTK::Variable %{

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        Variable p = (Variable)o;
        if (p == null) return false;
        return Utils.AreEqualVariable(this, p);
    }

    public boolean equals(Variable p) {
        if (p == null) return false;
        return Utils.AreEqualVariable(this, p);
    }

    //add back once GetHashValue is defined for SWIG and not just SWIGCSHARP
    /*@Override
    public int hashCode() {
        return (int)GetHashValue();
    }*/

%}

%typemap(javacode) CNTK::NDShape %{

    public java.util.ArrayList<Long> getDimensions(){
        java.util.ArrayList<Long> ret = new java.util.ArrayList<Long>((int)GetRank());
        for (int i = 0; i < GetDimensions().size(); ++i ) {
            ret.add((Long)GetDimensions().get(i));
        }
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        NDShape p = (NDShape)o;
        if (p == null) return false;
        return Utils.AreEqualShape(this, p);
    }

    public boolean equals(NDShape p) {
        if (p == null) return false;
        return Utils.AreEqualShape(this, p);
    }

    @Override
    public int hashCode() {
        return GetDimensions().hashCode();
    }

%}

%typemap(javacode) CNTK::Value %{
/*
 *    public static <T> Value CreateBatch(NDShape shape, java.util.ArrayList<T> batch, DeviceDescriptor device, boolean readOnly = false) {
 *        long shapeSize = shape.GetTotalSize();
 *
 *        if (batch.Count % shapeSize != 0)
 *            throw new ArgumentException("The number of elements in the batch must be a multiple of the size of the shape");
 *
 *        int count = batch.size() / shapeSize;
 *        var input = new java.util.ArrayList<java.util.ArrayList<T>>(count);
 *        for (int i = 0; i < count; i++)
 *        {
 *            java.util.ArrayList<T> seq = new java.util.ArrayList<T>();
 *            seq.addAll(batch.subList((int)(i * shapeSize), (int)shapeSize));
 *            input.Add(seq);
 *        }
 *        // Pass the empty seqStartFlags means all sequences have the start flag with true.
 *        return Create<T>(shape, input, new System.Collections.Generic.List<bool>(0), device, readOnly);
 *    }
 */
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

