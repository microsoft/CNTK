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

%include "CNTKManagedCommon.i"

%pragma(java) jniclasscode=%{
  static {
    String libName = "Cntk.Core.JavaBinding-2.0rc2";
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
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(DeviceDescriptor p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
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
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(Axis p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
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
    public static Function Load(byte[] modelBuffer, DeviceDescriptor computeDevice)
    {
        return Load(modelBuffer, (long)modelBuffer.length, computeDevice);
    }

    // TODO: look at C# implementation and make it look more like that
    private VariableVector argumentVector;
    private VariableVector outputVector;
    private java.util.ArrayList<Variable> argumentList;
    private java.util.ArrayList<Variable> outputList;

    private UnorderedMapVariableValuePtr outMap = new UnorderedMapVariableValuePtr();

    public java.util.List<Variable> getOutputs() {
        if (outputVector == null) {
            outputVector = GetOutputs();
            outputList = new java.util.ArrayList<Variable>((int)outputVector.size());
            for (int i = 0; i < outputVector.size(); ++i){
                outputList.add(outputVector.get(i));
            }
        }
        return outputList;
    }

    public java.util.List<Variable> getArguments() {
        if (argumentVector == null) {
            argumentVector = GetArguments();
            argumentList = new java.util.ArrayList<Variable>((int)argumentVector.size());
            for (int i = 0; i < argumentVector.size(); ++i){
                argumentList.add(argumentVector.get(i));
            }
        }
        return argumentList;
    }

    public static Function Combine(java.util.ArrayList<Variable> outputVariable) {
        VariableVector varVect = new VariableVector();
        for (int i = 0; i < outputVariable.size(); ++i)
        {
            varVect.add(varVect.get(i));
        }
        return CNTKLib.Combine(varVect);
    }
%}

%typemap(javacode) CNTK::Variable %{
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
        return (int)GetHashValue();
    }
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
        return CNTKLib.AreEqual(this, p);
    }

    public boolean equals(NDShape p) {
        if (p == null) return false;
        return CNTKLib.AreEqual(this, p);
    }

    @Override
    public int hashCode() {
        return GetDimensions().hashCode();
    }
%}

%typemap(javacode) CNTK::NDMask %{
%}

%typemap(javacode) CNTK::Value %{
%}

%typemap(javacode) CNTK::NDArrayView %{
%}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
