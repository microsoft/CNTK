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

    // Create Value object from dense input as batch data.
    public static Value createBatch(NDShape sampleShape, float[] batch, DeviceDescriptor device, boolean readOnly) {
        FloatVector inputVector = Helper.AsFloatVector(batch);
        return _CreateBatchFloat(sampleShape, inputVector, device, readOnly);
    }

    public static Value createBatch(NDShape sampleShape, float[] batch, DeviceDescriptor device) {
        return Value.createBatch(sampleShape, batch, device, false);
    }

    public static Value createBatch(NDShape sampleShape, double[] batch, DeviceDescriptor device, boolean readOnly) {
        DoubleVector inputVector = Helper.AsDoubleVector(batch);
        return _CreateBatchDouble(sampleShape, inputVector, device, readOnly);
    }

    public static Value createBatch(NDShape sampleShape, double[] batch, DeviceDescriptor device) {
        return createBatch(sampleShape, batch, device, false);
    }

    // Create Value object from dense input as sequence data.
    public static Value CreateSequence(NDShape sampleShape,
                                       float[] sequence,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        return CreateSequence(sampleShape, sequence, true, device, readOnly);
    }

    public static Value CreateSequence(NDShape sampleShape,
                                       float[] sequence,
                                       DeviceDescriptor device) {
        return CreateSequence(sampleShape, sequence, device, false);
    }

    // Create Value object from dense input as sequence data with sequenceStartFlag.
    public static Value CreateSequence(NDShape sampleShape,
                                       float[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        FloatVector inputVector = Helper.AsFloatVector(sequence);
        return _CreateSequenceFloat(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
    }

    public static Value CreateSequence(NDShape sampleShape,
                                       float[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device) {
        return CreateSequence(sampleShape, sequence, sequenceStartFlag, device, false);
    }

    public static Value CreateSequence(NDShape sampleShape,
                                       double[] sequence,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        return CreateSequence(sampleShape, sequence, true, device, readOnly);
    }

    public static Value CreateSequence(NDShape sampleShape,
                                       double[] sequence,
                                       DeviceDescriptor device) {
        return CreateSequence(sampleShape, sequence, device, false);
    }

    // Create Value object from dense input as sequence data with sequenceStartFlag.
    public static Value CreateSequence(NDShape sampleShape,
                                       double[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        DoubleVector inputVector = Helper.AsDoubleVector(sequence);
        return _CreateSequenceDouble(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
    }

    public static Value CreateSequence(NDShape sampleShape,
                                       double[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device) {
        return CreateSequence(sampleShape, sequence, sequenceStartFlag, device, false);
    }

    // Create Value object from dense input as batch of sequences data.
    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return Create(sampleShape, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return Create(sampleShape, batchOfSequences, new boolean[] {false}, device, false);
    }

    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return Create(sampleShape, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return Create(sampleShape, batchOfSequences, new boolean[] {false}, device, false);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device) {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, false);
    }

    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value CreateBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device) {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, false);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value Create(NDShape sampleShape,
                               float[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device,
                               boolean readOnly) {
        BoolVector seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        FloatVectorVector inputAsSequencesVector = new FloatVectorVector();
        for (float[] seq : sequences) {
            FloatVector seqVector = Helper.AsFloatVector(seq);
            // The seqVector is copied when adding to inputAsSequencesVector.
            inputAsSequencesVector.add(seqVector);
        }
        return _CreateDenseFloat(sampleShape, inputAsSequencesVector, seqFlags, device, readOnly);
    }

    public static Value Create(NDShape sampleShape,
                               float[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device) {
        return Create(sampleShape, sequences, sequenceStartFlags, device, false);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value Create(NDShape sampleShape,
                               double[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device,
                               boolean readOnly) {
        BoolVector seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        DoubleVectorVector inputAsSequencesVector = new DoubleVectorVector();
        for (double[] seq : sequences) {
            DoubleVector seqVector = Helper.AsDoubleVector(seq);
            // The seqVector is copied when adding to inputAsSequencesVector.
            inputAsSequencesVector.add(seqVector);
        }
        return _CreateDenseDouble(sampleShape, inputAsSequencesVector, seqFlags, device, readOnly);
    }

    public static Value Create(NDShape sampleShape,
                               double[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device) {
        return Create(sampleShape, sequences, sequenceStartFlags, device, false);
    }

    // Create Value object from OneHotVector input, for N-dimenstional tensor. Only Create() method for now.
    public static Value Create(NDShape sampleShape,
                               long[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device,
                               boolean readOnly) {
        BoolVector seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        SizeTVectorVector inputSeqVector = new SizeTVectorVector();
        for (long[] seq : sequences) {
            SizeTVector s = Helper.AsSizeTVector(seq);
            inputSeqVector.add(s);
        }
        return _CreateOneHotFloat(sampleShape, inputSeqVector, seqFlags, device, readOnly);
    }

    public static Value Create(NDShape sampleShape,
                               long[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device) {
        return Create(sampleShape, sequences, sequenceStartFlags, device, false);
    }

    // Create Value object from OneHotVector input as batch data, for 1D tensor only.
    public static Value CreateBatchFloat(long dimension, long[] batch, DeviceDescriptor device, boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(batch);
        return Value._CreateBatchFloat(dimension, inputVector, device, readOnly);
    }

    public static Value CreateBatchFloat(long dimension, long[] batch, DeviceDescriptor device) {
        return CreateBatchFloat(dimension, batch, device, false);
    }

    public static Value CreateBatchDouble(long dimension, long[] batch, DeviceDescriptor device, boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(batch);
        return Value._CreateBatchDouble(dimension, inputVector, device, readOnly);
    }

    public static Value CreateBatchDouble(long dimension, long[] batch, DeviceDescriptor device) {
        return CreateBatchDouble(dimension, batch, device, false);
    }

    // Create Value object from OneHotVector input as sequence data with sequenceStartFlag, for 1D tensor only.
    public static Value CreateSequenceFloat(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(sequence);
        return Value._CreateSequenceFloat(dimension, inputVector, sequenceStartFlag, device, readOnly);
    }

    // Create Value object from OneHotVector input as sequence data, for 1D tensor only.
    public static Value CreateSequenceFloat(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device,
                                          boolean readOnly) {
        return CreateSequenceFloat(dimension, sequence, true, device, readOnly);
    }

    public static Value CreateSequenceFloat(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device) {
        return CreateSequenceFloat(dimension, sequence, sequenceStartFlag, device, false);
    }

    public static Value CreateSequenceFloat(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device) {
        return CreateSequenceFloat(dimension, sequence, device, false);
    }

    public static Value CreateSequenceDouble(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(sequence);
        return Value._CreateSequenceDouble(dimension, inputVector, sequenceStartFlag, device, readOnly);
    }

    // Create Value object from OneHotVector input as sequence data, for 1D tensor only.
    public static Value CreateSequenceDouble(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device,
                                          boolean readOnly) {
        return CreateSequenceDouble(dimension, sequence, true, device, readOnly);
    }

    public static Value CreateSequenceDouble(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device) {
        return CreateSequenceDouble(dimension, sequence, sequenceStartFlag, device, false);
    }

    public static Value CreateSequenceDouble(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device) {
        return CreateSequenceDouble(dimension, sequence, device, false);
    }

    // Create Value object from OneHotVector input as batch of sequences data, for 1D tensor only.
    public static Value CreateBatchOfSequencesFloat(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return CreateFloat(dimension, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value CreateBatchOfSequencesFloat(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return CreateFloat(dimension, batchOfSequences, new boolean[] {false}, device, false);
    }

    public static Value CreateBatchOfSequencesDouble(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return CreateDouble(dimension, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value CreateBatchOfSequencesDouble(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return CreateDouble(dimension, batchOfSequences, new boolean[] {false}, device, false);
    }

    // Create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
    public static Value CreateBatchOfSequencesFloat(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device,
                                                   boolean readOnly) {
        return CreateFloat(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    public static Value CreateBatchOfSequencesFloat(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device) {
        return CreateFloat(dimension, batchOfSequences, sequenceStartFlags, device, false);
    }

    public static Value CreateBatchOfSequencesDouble(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device,
                                                   boolean readOnly) {
        return CreateDouble(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    public static Value CreateBatchOfSequencesDouble(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device) {
        return CreateDouble(dimension, batchOfSequences, sequenceStartFlags, device, false);
    }

    // Create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
    public static Value CreateFloat(long dimension,
                                    long[][] sequences,
                                    boolean[] sequenceStartFlags,
                                    DeviceDescriptor device,
                                    boolean readOnly) {
        BoolVector seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        SizeTVectorVector inputSeqVector = new SizeTVectorVector();
        for (long[] seq : sequences) {
            SizeTVector s = Helper.AsSizeTVector(seq);
            inputSeqVector.add(s);
        }
        return Value._CreateOneHotFloat(dimension, inputSeqVector, seqFlags, device, readOnly);
    }

    public static Value CreateFloat(long dimension,
                                    long[][] sequences,
                                    boolean[] sequenceStartFlags,
                                    DeviceDescriptor device) {
        return CreateFloat(dimension, sequences, sequenceStartFlags, device, false);
    }

    public static Value CreateDouble(long dimension,
                                     long[][] sequences,
                                     boolean[] sequenceStartFlags,
                                     DeviceDescriptor device,
                                     boolean readOnly) {
        BoolVector seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        SizeTVectorVector inputSeqVector = new SizeTVectorVector();
        for (long[] seq : sequences) {
            SizeTVector s = Helper.AsSizeTVector(seq);
            inputSeqVector.add(s);
        }
        return Value._CreateOneHotDouble(dimension, inputSeqVector, seqFlags, device, readOnly);
    }

    public static Value CreateDouble(long dimension,
                                     long[][] sequences,
                                     boolean[] sequenceStartFlags,
                                     DeviceDescriptor device) {
        return CreateDouble(dimension, sequences, sequenceStartFlags, device, false);
    }

%}

%typemap(javacode) CNTK::NDArrayView %{
%}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
