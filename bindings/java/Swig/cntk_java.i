//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cntk_java.i -- SWIG Interface file for Java
//

//JNI defines UNUSED macro as well, undefining it so it doesn't conflict with CNTK's
%{
#undef UNUSED
%}

%{
    #pragma warning(disable : 4267) //warning C4267: 'initializing': conversion from 'size_t' to 'jsize', possible loss of data
%}

%include "CNTKManagedCommon.i"

%pragma(java) jniclasscode=%{
  static {
      CNTKNativeUtils.loadAll();
  }
%}

// Java specific extention.
%typemap(javacode) CNTK::DeviceDescriptor %{

    public java.util.List<DeviceDescriptor> getAllDevices() {
        DeviceDescriptorVector devices = _AllDevices();
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
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(DeviceDescriptor p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    @Override
    public int hashCode() {
        return getType().hashCode();
    }
%}

%typemap(javacode) CNTK::Axis %{

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        Axis p = (Axis)o;
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(Axis p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    @Override
    public int hashCode() {
        if (this.isDynamicAxis()) {
            return getName().hashCode();
        } else {
            return this.getStaticAxisIndex();
        }
    }
%}


%typemap(javacode) CNTK::Function %{
    private FunctionPtrVector ref;
    public void addReference(FunctionPtrVector fpv) {
        ref = fpv;
    }

    public static Function load(byte[] modelBuffer, DeviceDescriptor computeDevice)
    {
        return load(modelBuffer, (long)modelBuffer.length, computeDevice);
    }

    public java.util.List<Variable> getInputs() {
        VariableVector inputVector = _Inputs();
        java.util.ArrayList<Variable> inputList = new java.util.ArrayList<Variable>((int)inputVector.size());
        for (int i = 0; i < inputVector.size(); ++i){
            Variable var = inputVector.get(i);
            var.addReference(inputVector);
            inputList.add(var);
        }
        return inputList;
    }

    public java.util.List<Variable> getOutputs() {
        VariableVector outputVector = _Outputs();
        java.util.ArrayList<Variable> outputList = new java.util.ArrayList<Variable>((int)outputVector.size());
        for (int i = 0; i < outputVector.size(); ++i){
            Variable var = outputVector.get(i);
            var.addReference(outputVector);
            outputList.add(var);
        }
        return outputList;
    }

    public java.util.List<Variable> getArguments() {
        VariableVector argumentVector = _Arguments();
        java.util.ArrayList<Variable> argumentList = new java.util.ArrayList<Variable>((int)argumentVector.size());
        for (int i = 0; i < argumentVector.size(); ++i){
            Variable var = argumentVector.get(i);
            var.addReference(argumentVector);
            argumentList.add(var);
        }
        return argumentList;
    }

    public java.util.List<Function> findAllWithName(String x) {
        FunctionPtrVector functionVector = _FindAllWithName(x);
        java.util.ArrayList<Function> functionList = new java.util.ArrayList<Function>((int)functionVector.size());
        for (int i = 0; i < functionVector.size(); ++i){
            Function func = functionVector.get(i);
            func.addReference(functionVector);
            functionList.add(func);
        }
        return functionList;
    }

    public static Function combine(java.util.ArrayList<Variable> outputVariable) {
        VariableVector varVect = new VariableVector();
        for (int i = 0; i < outputVariable.size(); ++i)
        {
            varVect.add(outputVariable.get(i));
        }
        return CNTKLib.Combine(varVect);
    }
%}

%typemap(javacode) CNTK::Variable %{
    private VariableVector ref;
    public void addReference(VariableVector vv) {
        ref = vv;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        Variable p = (Variable)o;
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(Variable p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    @Override
    public int hashCode() {
        return (int)_GetHashValue();
    }
%}

%typemap(javacode) CNTK::NDShape %{

    public long[] getDimensions(){
        SizeTVector dimensionVector = _Dimensions();
        long[] ret = new long[(int)getRank()];
        for (int i = 0; i < dimensionVector.size(); ++i ) {
            ret[i] = dimensionVector.get(i);
        }
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        NDShape p = (NDShape)o;
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(NDShape p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    @Override
    public int hashCode() {
        return _Dimensions().hashCode();
    }
%}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
