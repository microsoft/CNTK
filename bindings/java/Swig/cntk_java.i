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

    // creates a new Function from specified operands.
    public static Function combine(java.util.ArrayList<Variable> outputVariable) {
        VariableVector varVect = new VariableVector();
        for (int i = 0; i < outputVariable.size(); ++i) {
            varVect.add(outputVariable.get(i));
        }
        return CNTKLib.Combine(varVect);
    }

    // creates a composite function from the rootFunction.
    public static Function asComposite(Function rootFunction, String name) {
        return CNTKLib.AsComposite(rootFunction, name);
    }

    public static Function asComposite(Function rootFunction) {
        return CNTKLib.AsComposite(rootFunction, "");
    }

    // create a new Function which is the alias of operand.
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

    // creates a new NDShape.
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

    // create Value object from dense input as batch data.
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

    // create Value object from dense input as sequence data.
    public static Value createSequence(NDShape sampleShape,
                                       float[] sequence,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        return createSequence(sampleShape, sequence, true, device, readOnly);
    }

    public static Value createSequence(NDShape sampleShape,
                                       float[] sequence,
                                       DeviceDescriptor device) {
        return createSequence(sampleShape, sequence, device, false);
    }

    // create Value object from dense input as sequence data with sequenceStartFlag.
    public static Value createSequence(NDShape sampleShape,
                                       float[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        FloatVector inputVector = Helper.AsFloatVector(sequence);
        return _CreateSequenceFloat(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
    }

    public static Value createSequence(NDShape sampleShape,
                                       float[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device) {
        return createSequence(sampleShape, sequence, sequenceStartFlag, device, false);
    }

    public static Value createSequence(NDShape sampleShape,
                                       double[] sequence,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        return createSequence(sampleShape, sequence, true, device, readOnly);
    }

    public static Value createSequence(NDShape sampleShape,
                                       double[] sequence,
                                       DeviceDescriptor device) {
        return createSequence(sampleShape, sequence, device, false);
    }

    // create Value object from dense input as sequence data with sequenceStartFlag.
    public static Value createSequence(NDShape sampleShape,
                                       double[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device,
                                       boolean readOnly) {
        DoubleVector inputVector = Helper.AsDoubleVector(sequence);
        return _CreateSequenceDouble(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
    }

    public static Value createSequence(NDShape sampleShape,
                                       double[] sequence,
                                       boolean sequenceStartFlag,
                                       DeviceDescriptor device) {
        return createSequence(sampleShape, sequence, sequenceStartFlag, device, false);
    }

    // create Value object from dense input as batch of sequences data.
    public static Value createBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return create(sampleShape, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value createBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return create(sampleShape, batchOfSequences, new boolean[] {false}, device, false);
    }

    public static Value createBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return create(sampleShape, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value createBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return create(sampleShape, batchOfSequences, new boolean[] {false}, device, false);
    }

    // create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value createBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    // create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value createBatchOfSequences(NDShape sampleShape,
                                               float[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device) {
        return create(sampleShape, batchOfSequences, sequenceStartFlags, device, false);
    }

    public static Value createBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    // create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value createBatchOfSequences(NDShape sampleShape,
                                               double[][] batchOfSequences,
                                               boolean[] sequenceStartFlags,
                                               DeviceDescriptor device) {
        return create(sampleShape, batchOfSequences, sequenceStartFlags, device, false);
    }

    // create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value create(NDShape sampleShape,
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

    public static Value create(NDShape sampleShape,
                               float[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device) {
        return create(sampleShape, sequences, sequenceStartFlags, device, false);
    }

    // create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value create(NDShape sampleShape,
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

    public static Value create(NDShape sampleShape,
                               double[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device) {
        return create(sampleShape, sequences, sequenceStartFlags, device, false);
    }

    // create Value object from OneHotVector input, for N-dimenstional tensor. Only create() method for now.
    public static Value create(NDShape sampleShape,
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

    public static Value create(NDShape sampleShape,
                               long[][] sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device) {
        return create(sampleShape, sequences, sequenceStartFlags, device, false);
    }

    // create Value object from OneHotVector input as batch data, for 1D tensor only.
    public static Value createBatchFloat(long dimension, long[] batch, DeviceDescriptor device, boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(batch);
        return Value._CreateBatchFloat(dimension, inputVector, device, readOnly);
    }

    public static Value createBatchFloat(long dimension, long[] batch, DeviceDescriptor device) {
        return createBatchFloat(dimension, batch, device, false);
    }

    public static Value createBatchDouble(long dimension, long[] batch, DeviceDescriptor device, boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(batch);
        return Value._CreateBatchDouble(dimension, inputVector, device, readOnly);
    }

    public static Value createBatchDouble(long dimension, long[] batch, DeviceDescriptor device) {
        return createBatchDouble(dimension, batch, device, false);
    }

    // create Value object from OneHotVector input as sequence data with sequenceStartFlag, for 1D tensor only.
    public static Value createSequenceFloat(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(sequence);
        return Value._CreateSequenceFloat(dimension, inputVector, sequenceStartFlag, device, readOnly);
    }

    // create Value object from OneHotVector input as sequence data, for 1D tensor only.
    public static Value createSequenceFloat(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device,
                                          boolean readOnly) {
        return createSequenceFloat(dimension, sequence, true, device, readOnly);
    }

    public static Value createSequenceFloat(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device) {
        return createSequenceFloat(dimension, sequence, sequenceStartFlag, device, false);
    }

    public static Value createSequenceFloat(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device) {
        return createSequenceFloat(dimension, sequence, device, false);
    }

    public static Value createSequenceDouble(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        SizeTVector inputVector = Helper.AsSizeTVector(sequence);
        return Value._CreateSequenceDouble(dimension, inputVector, sequenceStartFlag, device, readOnly);
    }

    // create Value object from OneHotVector input as sequence data, for 1D tensor only.
    public static Value createSequenceDouble(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device,
                                          boolean readOnly) {
        return createSequenceDouble(dimension, sequence, true, device, readOnly);
    }

    public static Value createSequenceDouble(long dimension,
                                            long[] sequence,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device) {
        return createSequenceDouble(dimension, sequence, sequenceStartFlag, device, false);
    }

    public static Value createSequenceDouble(long dimension,
                                          long[] sequence,
                                          DeviceDescriptor device) {
        return createSequenceDouble(dimension, sequence, device, false);
    }

    // create Value object from OneHotVector input as batch of sequences data, for 1D tensor only.
    public static Value createBatchOfSequencesFloat(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return createFloat(dimension, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value createBatchOfSequencesFloat(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return createFloat(dimension, batchOfSequences, new boolean[] {false}, device, false);
    }

    public static Value createBatchOfSequencesDouble(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device,
                                               boolean readOnly) {
        return createDouble(dimension, batchOfSequences, new boolean[] {false}, device, readOnly);
    }

    public static Value createBatchOfSequencesDouble(long dimension,
                                               long[][] batchOfSequences,
                                               DeviceDescriptor device) {
        return createDouble(dimension, batchOfSequences, new boolean[] {false}, device, false);
    }

    // create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
    public static Value createBatchOfSequencesFloat(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device,
                                                   boolean readOnly) {
        return createFloat(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    public static Value createBatchOfSequencesFloat(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device) {
        return createFloat(dimension, batchOfSequences, sequenceStartFlags, device, false);
    }

    public static Value createBatchOfSequencesDouble(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device,
                                                   boolean readOnly) {
        return createDouble(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    public static Value createBatchOfSequencesDouble(long dimension,
                                                   long[][] batchOfSequences,
                                                   boolean[] sequenceStartFlags,
                                                   DeviceDescriptor device) {
        return createDouble(dimension, batchOfSequences, sequenceStartFlags, device, false);
    }

    // create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
    public static Value createFloat(long dimension,
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

    public static Value createFloat(long dimension,
                                    long[][] sequences,
                                    boolean[] sequenceStartFlags,
                                    DeviceDescriptor device) {
        return createFloat(dimension, sequences, sequenceStartFlags, device, false);
    }

    public static Value createDouble(long dimension,
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

    public static Value createDouble(long dimension,
                                     long[][] sequences,
                                     boolean[] sequenceStartFlags,
                                     DeviceDescriptor device) {
        return createDouble(dimension, sequences, sequenceStartFlags, device, false);
    }

    // create Value object from sparse input as sequence data with sequenceStartFlag, for N-dimensional tensor. Only createSequence() for now.
    public static Value createSequenceFloat(NDShape sampleShape, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, float[] nonZeroValues,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        if (nonZeroValues.length != rowIndices.length) {
            throw new java.lang.IllegalArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (colStarts.length != sequenceLength + 1) {
            throw new java.lang.IllegalArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
        }
        long numNonZeroValues = (long)nonZeroValues.length;

        return Value._CreateSequenceFloat(sampleShape, (long)sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
    }

    public static Value createSequenceFloat(NDShape sampleShape, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, float[] nonZeroValues,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device) {
        return createSequenceFloat(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, sequenceStartFlag, device, false);
    }

    // create Value object from sparse input as sequence data, for 1D tensor. Only createSequence() for now.
    public static Value createSequenceFloat(int dimension, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, float[] nonZeroValues,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        if (nonZeroValues.length != rowIndices.length) {
            throw new java.lang.IllegalArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (colStarts.length != sequenceLength + 1) {
            throw new java.lang.IllegalArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
        }
        long numNonZeroValues = (long)nonZeroValues.length;

        return Value._CreateSequenceFloat(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, true, device, readOnly);
    }

    public static Value createSequenceFloat(int dimension, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, float[] nonZeroValues,
                                            DeviceDescriptor device) {
        return createSequenceFloat(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, device, false);
    }

    public static Value createSequenceDouble(NDShape sampleShape, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, double[] nonZeroValues,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        if (nonZeroValues.length != rowIndices.length) {
            throw new java.lang.IllegalArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (colStarts.length != sequenceLength + 1) {
            throw new java.lang.IllegalArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
        }
        long numNonZeroValues = (long)nonZeroValues.length;

        return Value._CreateSequenceDouble(sampleShape, (long)sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
    }

    public static Value createSequenceDouble(NDShape sampleShape, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, double[] nonZeroValues,
                                            boolean sequenceStartFlag,
                                            DeviceDescriptor device) {
        return createSequenceDouble(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, sequenceStartFlag, device, false);
    }

    // create Value object from sparse input as sequence data, for 1D tensor. Only createSequence() for now.
    public static Value createSequenceDouble(int dimension, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, double[] nonZeroValues,
                                            DeviceDescriptor device,
                                            boolean readOnly) {
        if (nonZeroValues.length != rowIndices.length) {
            throw new java.lang.IllegalArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
        }
        if (colStarts.length != sequenceLength + 1) {
            throw new java.lang.IllegalArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
        }
        long numNonZeroValues = (long)nonZeroValues.length;

        return Value._CreateSequenceDouble(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, true, device, readOnly);
    }

    public static Value createSequenceDouble(int dimension, int sequenceLength,
                                            int[] colStarts, int[] rowIndices, double[] nonZeroValues,
                                            DeviceDescriptor device) {
        return createSequenceDouble(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, device, false);
    }

    // create Value object from NDArrayViews.
    public static Value create(NDShape sampleShape,
                               java.lang.Iterable<NDArrayView> sequences,
                               DeviceDescriptor device,
                               boolean readOnly) {
        return create(sampleShape, sequences, new boolean[] {false}, device, readOnly);
    }

    // create Value object from NDArrayViews with sequenceStartFlags
    public static Value create(NDShape sampleShape,
                               java.lang.Iterable<NDArrayView> sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device,
                               boolean readOnly) {
        return create(sampleShape, sequences, sequenceStartFlags, device, readOnly, false);
    }

    // create Value object from NDArrayViews with sequenceStartFlags
    public static Value create(NDShape sampleShape,
                               java.lang.Iterable<NDArrayView> sequences,
                               boolean[] sequenceStartFlags,
                               DeviceDescriptor device,
                               boolean readOnly,
                               boolean createNewCopy) {
        NDArrayViewPtrVector seqVector = new NDArrayViewPtrVector();
        for (NDArrayView element : sequences) {
            seqVector.add(element);
        }
        BoolVector startFlags = Helper.AsBoolVector(sequenceStartFlags);
        return create(sampleShape, seqVector, startFlags, device, readOnly, createNewCopy);
    }

    //
    // Return the data of the Value object as a list of sequences with variable length.
    // This method returns an IList<IList<T>>. Each element of the outer list represents a sequence.
    // Each sequence, represented by IList<T>, contains a variable number of samples.
    // Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the variable shape.
    // The number of samples = (the count of elements in IList<T>)/(the count of elements of the sample)
    // The shape of the variable should match the shape of the Value object.
    //
    public float[][] getDenseDataFloat(Variable outputVariable) {
        float[][] sequences = new float[][] {{}};
        if (getDataType() != DataType.Float) {
            throw new java.lang.IllegalArgumentException("The value type does not match the list type.");
        }

        FloatVectorVector seqVec = new FloatVectorVector();
        _CopyVariableValueToFloat(outputVariable, seqVec);

        for (int i = 0; i < seqVec.size(); ++i) {
            FloatVector innerVec = seqVec.get(i);

            float[] valArray = new float[(int)innerVec.size()];
            for (int j = 0; j < innerVec.size(); ++j) {
                valArray[j] = innerVec.get(j);
            }

            sequences[i] = valArray;
        }
        return sequences;
    }

    public double[][] getDenseDataDouble(Variable outputVariable) {
        double[][] sequences = new double[][] {{}};
        if (getDataType() != DataType.Double) {
            throw new java.lang.IllegalArgumentException("The value type does not match the list type.");
        }

        DoubleVectorVector seqVec = new DoubleVectorVector();
        _CopyVariableValueToDouble(outputVariable, seqVec);

        for (int i = 0; i < seqVec.size(); ++i) {
            DoubleVector innerVec = seqVec.get(i);

            double[] valArray = new double[(int)innerVec.size()];
            for (int j = 0; j < innerVec.size(); ++j) {
                valArray[j] = innerVec.get(j);
            }

            sequences[i] = valArray;
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
    public int[][] GetOneHotData(Variable outputVariable) {
        int[][] sequences = new int[][] {{}};
        SizeTVectorVector seqVec = new SizeTVectorVector();
        _CopyVariableValueTo(outputVariable, seqVec);
        for (int i = 0; i < seqVec.size(); ++i) {
            SizeTVector innerVec = seqVec.get(i);

            int[] valArray = new int[(int)innerVec.size()];
            for (int j = 0; j < innerVec.size(); ++j) {
                valArray[j] = (int)innerVec.get(j);
            }

            sequences[i] = valArray;
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
     /*
      *@deprecated CopyVariableValueToFloat() will be deprecated soon. Please use GetDenseData() instead.
      */
    @Deprecated
    public void CopyVariableValueToFloat(Variable outputVariable, java.util.List<java.util.List<Float>> sequences) {
        sequences.clear();
        if (getDataType() != DataType.Float) {
            throw new java.lang.IllegalArgumentException("The value type does not match the list type.");
        }

        FloatVectorVector seqVecVec = new FloatVectorVector();
        _CopyVariableValueToFloat(outputVariable, seqVecVec);

        for (int i = 0; i < seqVecVec.size(); ++i) {
            FloatVector seqVec = seqVecVec.get(i);
            java.util.List<Float> seqList = new java.util.ArrayList<>();
            for (int j = 0; j < seqVec.size(); ++j) {
                seqList.add(seqVec.get(j));
            }
            sequences.add(new java.util.ArrayList<Float>(seqList));
        }
    }

     /*
      *@deprecated CopyVariableValueToDouble() will be deprecated soon. Please use GetDenseData() instead.
      */
    @Deprecated
    public void CopyVariableValueToDouble(Variable outputVariable, java.util.List<java.util.List<Double>> sequences) {
        sequences.clear();
        if (getDataType() != DataType.Double) {
            throw new java.lang.IllegalArgumentException("The value type does not match the list type.");
        }

        DoubleVectorVector seqVecVec = new DoubleVectorVector();
        _CopyVariableValueToDouble(outputVariable, seqVecVec);

        for (int i = 0; i < seqVecVec.size(); ++i) {
            DoubleVector seqVec = seqVecVec.get(i);
            java.util.List<Double> seqList = new java.util.ArrayList<>();
            for (int j = 0; j < seqVec.size(); ++j) {
                seqList.add(seqVec.get(j));
            }
            sequences.add(new java.util.ArrayList<Double>(seqList));
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
    public void CopyVariableValueToInt(Variable outputVariable, java.util.List<java.util.List<Integer>> sequences) {
        sequences.clear();

        SizeTVectorVector seqVecVec = new SizeTVectorVector();
        _CopyVariableValueTo(outputVariable, seqVecVec);

        for (int i = 0; i < seqVecVec.size(); ++i) {
            SizeTVector seqVec = seqVecVec.get(i);
            java.util.List<Integer> seqList = new java.util.ArrayList<>();
            for (int j = 0; j < seqVec.size(); ++j) {
                seqList.add((int)seqVec.get(j));
            }
            sequences.add(new java.util.ArrayList<Integer>(seqList));
        }
    }

    /*public void GetSparseData<T>( Variable outputVariable,*/
                                    /*out int sequenceLength,*/
                                    /*out System.Collections.Generic.IList<int> colStarts,*/
                                    /*out System.Collections.Generic.IList<int> rowIndices,*/
                                    /*out System.Collections.Generic.IList<T> nonZeroValues,*/
                                    /*out int numNonZeroValues) */
    /*{*/
        /*var colStartVec = new IntVector();*/
        /*var rowIndicesVec = new IntVector();*/

        /*int[] n1 = new int[1];*/
        /*int[] n2 = new int[1];*/

        /*if (typeof(T).Equals(typeof(float)))*/
        /*{*/
            /*if (getDataType() != DataType.Float)*/
            /*{*/
                /*throw new System.ArgumentException("The value type does not match the list type.");*/
            /*}*/

            /*var nonZeroValuesVec = new FloatVector();*/
            /*_CopyVariableValueToFloat(outputVariable, n1,  colStartVec,   */
                /*rowIndicesVec, nonZeroValuesVec, n2);*/
            /*nonZeroValues = nonZeroValuesVec as System.Collections.Generic.IList<T>;*/
        /*}*/
        /*else if (typeof(T).Equals(typeof(double)))*/
        /*{*/
            /*if (getDataType() != DataType.Double)*/
            /*{*/
                /*throw new System.ArgumentException("The value type does not match the list type.");*/
            /*}*/

            /*var nonZeroValuesVec = new DoubleVector();*/
            /*_CopyVariableValueToDouble(outputVariable, n1, colStartVec,*/
                /*rowIndicesVec, nonZeroValuesVec, n2);*/
            /*nonZeroValues = nonZeroValuesVec as System.Collections.Generic.IList<T>;*/
        /*}*/
        /*else*/
        /*{*/
            /*throw new System.ArgumentException("The value type does not match the list type.");*/
        /*}*/

        /*sequenceLength = n1[0];*/
        /*numNonZeroValues = n2[0];*/
        /*colStarts = colStartVec;*/
        /*rowIndices = rowIndicesVec;*/
    /*}*/


    /*// creates a new Value which is an alias of this Value.*/
    /*public Value Alias(boolean readOnly = false)*/
    /*{*/
        /*return _Alias(readOnly);*/
    /*}*/

%}

%typemap(javacode) CNTK::NDArrayView %{
    /*// Constructor using float dense input.*/
    /*public NDArrayView(NDShape viewShape, float[] dataBuffer, DeviceDescriptor device, boolean readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.length, device, readOnly)*/
    /*{*/
    /*}*/

    /*// Constructor using double dense input.*/
    /*public NDArrayView(NDShape viewShape, double[] dataBuffer, DeviceDescriptor device, boolean readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.length, device, readOnly)*/
    /*{*/
    /*}*/

    /*// Constructor using float sparse input.*/
    /*public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, float[] nonZeroValues, DeviceDescriptor device, boolean readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.length, device, readOnly)*/
    /*{*/
        /*if (rowIndices.length != nonZeroValues.length)*/
        /*{*/
            /*throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");*/
        /*}*/
        /*if (viewShape[viewShape.Rank-1] + 1 != colStarts.length)*/
        /*{*/
            /*throw new System.ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");*/
        /*}*/
    /*}*/

    /*// Constructor using double sparse input.*/
    /*public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, double[] nonZeroValues, DeviceDescriptor device, boolean readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.length, device, readOnly)*/
    /*{*/
        /*if (rowIndices.length != nonZeroValues.length)*/
        /*{*/
            /*throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");*/
        /*}*/
        /*if (viewShape[viewShape.Rank-1] + 1 != colStarts.length)*/
        /*{*/
            /*throw new System.ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");*/
        /*}*/
    /*}*/

    /*// Property Device.*/
    /*public DeviceDescriptor Device*/
    /*{*/
        /*get { return _Device(); }*/
    /*}*/

    /*// Property DataType.*/
    /*public DataType DataType*/
    /*{*/
        /*get { return getDataType(); }*/
    /*}*/

    /*// Property Shape.*/
    /*public NDShape Shape*/
    /*{*/
        /*get { return _Shape(); }*/
    /*}*/

    /*// Property StorageFormat.*/
    /*public StorageFormat StorageFormat*/
    /*{*/
        /*get { return _GetStorageFormat(); }*/
    /*}*/

    /*// Property IsSparse.*/
    /*public boolean IsSparse*/
    /*{*/
        /*get { return _IsSparse(); }*/
    /*}*/

    /*// Property IsReadOnly.*/
    /*public boolean IsReadOnly*/
    /*{*/
        /*get { return _IsReadOnly(); }*/
    /*}*/

    /*// Returns a slice view.*/
    /*public NDArrayView SliceView(System.Collections.Generic.IEnumerable<int> startOffset, System.Collections.Generic.IEnumerable<int> extent, boolean readOnly = false)*/
    /*{*/
        /*var startOffsetVector = Helper.AsSizeTVector(startOffset);*/

        /*var extentVector = Helper.AsSizeTVector(extent);*/

        /*return _SliceView(startOffsetVector, extentVector, readOnly);*/
    /*}*/

    /*// creates a new NDArrayView which is an alias of this NDArrayView.*/
    /*public NDArrayView Alias(boolean readOnly = false)*/
    /*{*/
        /*return _Alias(readOnly);*/
    /*}*/
%}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
