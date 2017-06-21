//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file : the project root for full license information.
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
    String libName = "Cntk.Core.JavaBinding-2.1";
    try {
       System.loadLibrary(libName);
    } catch (UnsatisfiedLinkError e) {
       try {
           System.loadLibrary(libName+'d');
       } catch (UnsatisfiedLinkError e2) {
          System.err.println("Native code library failed to load. \n" + e2);
          System.exit(1);
       }
    }
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

    // Set devices to be excluded.
    public static void setExcludedDevices(java.util.ArrayList<DeviceDescriptor> excluded) {
        DeviceDescriptorVector excludeVector = new DeviceDescriptorVector();
        for (DeviceDescriptor element : excluded)
        {
            excludeVector.add(element);
        }
        _SetExcludedDevices(excludeVector);
    }
%}

%typemap(javacode) CNTK::Axis %{

    private AxisVector avRef;
    public void addReference(AxisVector av) {
        avRef = av;
    }

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

    public static Function load(byte[] modelBuffer, DeviceDescriptor computeDevice) {
        return load(modelBuffer, (long)modelBuffer.length, computeDevice);
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


    // Evaluates the Function using provided inputs.
    public void evaluate(java.util.Map<Variable, Value> inputs, java.util.Map<Variable, Value> outputs, DeviceDescriptor computeDevice) {
        // Evaluate the rootFunction.
        UnorderedMapVariableValuePtr inMap = new UnorderedMapVariableValuePtr();
        UnorderedMapVariableValuePtr outMap = new UnorderedMapVariableValuePtr();

        for (java.util.Map.Entry<Variable, Value> p : inputs.entrySet()) {
            inMap.put(p.getKey(), p.getValue());
        }

        for (java.util.Map.Entry<Variable, Value> p : outputs.entrySet()) {
            outMap.put(p.getKey(), p.getValue());
        }

        _Evaluate(inMap, outMap, computeDevice);

        for (java.util.Map.Entry<Variable, Value> p : outMap.entrySet()) {
            outputs.put(p.getKey(), p.getValue());
        }
    }

    // Creates a new Function from specified operands.
    public static Function combine(java.util.ArrayList<Variable> outputVariable) {
        VariableVector varVect = new VariableVector();
        for (int i = 0; i < outputVariable.size(); ++i) {
            varVect.add(varVect.get(i));
        }
        return CNTKLib.Combine(varVect);
    }

    // Creates a composite function from the rootFunction.
    public static Function asComposite(Function rootFunction, String name) {
        return CNTKLib.AsComposite(rootFunction, name);
    }

    public static Function asComposite(Function rootFunction) {
        return CNTKLib.AsComposite(rootFunction, "");
    }

    // Create a new Function which is the alias of operand.
    public static Function alias(Variable operand, String name) {
        return CNTKLib.Alias(operand, name);
    }

    public static Function alias(Variable operand) {
        return CNTKLib.Alias(operand, "");
    }
%}

%typemap(javacode) CNTK::Variable %{
    private VariableVector vvRef;
    public void addReference(VariableVector vv) {
        vvRef = vv;
    }

    // Property DynamicAxes.
    public java.util.List<Axis> getDynamicAxes() {
        AxisVector axisVector = _DynamicAxes();
        java.util.ArrayList<Axis> axisList = new java.util.ArrayList<Axis>((int)axisVector.size());
        for (int i = 0; i < axisVector.size(); ++i){
            Axis axis = axisVector.get(i);
            axis.addReference(axisVector);
            axisList.add(axis);
        }
        return axisList;
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

    // Creates a new NDShape.
    public static NDShape createNDShape(long[] dimensions) {
        SizeTVector dimVector = new SizeTVector();
        for (long element : dimensions) {
            if (element < 0) {
                throw new java.lang.IllegalArgumentException("The parameter dimensions cannot contain a negative value");
            }
            dimVector.add(element);
        }
        return new NDShape(dimVector);
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

%typemap(javacode) CNTK::NDMask %{

    // Invidates a section of a NDShape.
    public void InvalidateSection(long[] sectionOffset, NDShape sectionShape) {
        SizeTVector offsetVector = Helper.AsSizeTVector(sectionOffset);
        _InvalidateSection(offsetVector, sectionShape);
    }

    // Marks sequence begin.
    public void MarkSequenceBegin(long[] offset) {
        SizeTVector offsetVector = Helper.AsSizeTVector(offset);
        _MarkSequenceBegin(offsetVector);
    }

    // Marks sequence begins : a NDShape.
    public void MarkSequenceBegin(long[] offset, NDShape sectionShape) {
        SizeTVector offsetVector = Helper.AsSizeTVector(offset);
        _MarkSequenceBegin(offsetVector, sectionShape);
    }
%}

%typemap(javacode) CNTK::Value %{
%}

%typemap(javacode) CNTK::NDArrayView %{
%}


%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
