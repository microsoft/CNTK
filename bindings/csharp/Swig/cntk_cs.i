//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cntk_cs.i -- SWIG Interface file for C#
//

%include "CNTKManagedCommon.i"

// C# specific extenstion
%typemap(cscode) CNTK::DeviceDescriptor %{

    // Property Id.
    public int Id
    {
        get { return (int)_Id(); }
    }

    // Property Type.
    public DeviceKind Type
    {
        get { return _Type(); }
    }

    // Property CPUDevice.
    public static DeviceDescriptor CPUDevice
    {
        get { return _CPUDevice(); }
    }

    // Returns the GPUDevice with the specific deviceId.
    public static DeviceDescriptor GPUDevice(int deviceId)
    {
        if (deviceId < 0)
        {
            throw new System.ArgumentException("The paraemter deviceId should not be a negative value");
        }
        return _GPUDevice((uint)deviceId);
    }

    // Gets all devices.
    public static System.Collections.Generic.IList<DeviceDescriptor> AllDevices()
    {
        var deviceVector = _AllDevices();
        // The CopyTo is to ensure the elements in the deviceVector can live beyond deviceVector itself.
        var deviceArray = new DeviceDescriptor[deviceVector.Count];
        deviceVector.CopyTo(deviceArray);
        var deviceList = new System.Collections.Generic.List<DeviceDescriptor>(deviceArray);
        return deviceList;
    }

    // Value equality.
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
        return CNTKLib.AreEqual(this, p);
    }

    // Value equality.
    public bool Equals(DeviceDescriptor p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Returns hash code value.
    public override int GetHashCode()
    {
        return this._Type().GetHashCode();
    }

    // Set devices to be excluded.
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


%typemap(cscode) CNTK::Axis %{

    // Property Name.
    public string Name
    {
        get { return _Name(); }
    }

    // Property IsStatic.
    public bool IsStatic
    {
        get { return _IsStaticAxis(); }
    }

    // Property IsDynamic.
    public bool IsDynamic
    {
        get { return _IsDynamicAxis(); }
    }

    // Property IsOrdered.
    public bool IsOrdered
    {
        get { return _IsOrdered(); }
    }

    // Returns index of this Axis.
    public int StaticAxisIndex(bool checkStaticAxis = true)
    {
        return _StaticAxisIndex(checkStaticAxis);
    }

    // Value equality.
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
        return CNTKLib.AreEqual(this, p);
    }

    // Value equality.
    public bool Equals(Axis p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Returns hash code value.
    public override int GetHashCode()
    {
        if (this._IsDynamicAxis())
        {
            return this.Name.GetHashCode();
        }
        else
        {
            return this.StaticAxisIndex(false).GetHashCode();
        }
    }
%}

%typemap(cscode) CNTK::Function %{

    // Property Name.
    public string Name
    {
        get { return _Name(); }
    }

    // Property Uid.
    public string Uid
    {
        get { return _Uid(); }
    }

    // Property RootFunction.
    public Function RootFunction
    {
        get { return _RootFunction(); }
    }

    // Property Outputs
    public System.Collections.Generic.IList<Variable> Outputs
    {
        get {
            var varVector = _Outputs();
            var varArray = new Variable[varVector.Count];
            // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
            varVector.CopyTo(varArray);
            var varList = new System.Collections.Generic.List<Variable>(varArray);
            return varList;
        }
    }

    // Property Output.
    public Variable Output
    {
        get { return _Output(); }
    }

    // Property OpName.
    public string OpName
    {
        get { return _OpName(); }
    }

    // Property IsComposite.
    public bool IsComposite
    {
        get { return _IsComposite(); }
    }

    // Property IsPrimitive.
    public bool IsPrimitive
    {
        get { return _IsPrimitive(); }
    }

    // Property IsBlock.
    public bool IsBlock
    {
        get { return _IsBlock(); }
    }

    // Property CurrentVersion
    public int CurrentVersion
    {
        get { return (int)_CurrentVersion();}
    }

    // Property Arguments.
    public System.Collections.Generic.IList<Variable> Arguments
    {
        get {
            var varVector = _Arguments();
            var varArray = new Variable[varVector.Count];
            // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
            varVector.CopyTo(varArray);
            var varList = new System.Collections.Generic.List<Variable>(varArray);
            return varList;
        }
    }

    // Property Inputs.
    public System.Collections.Generic.IList<Variable> Inputs
    {
        get {
            var varVector = _Inputs();
            var varArray = new Variable[varVector.Count];
            // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
            varVector.CopyTo(varArray);
            var varList = new System.Collections.Generic.List<Variable>(varArray);
            return varList;
        }
    }

    // Creates a new cloned function instance. For C# Eval, default ParameterCloningMethod is share.
    public Function Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod.Share)
    {
        return _Clone(ParameterCloningMethod.Share);
    }

    // Evaluates the Function using provided inputs.
    public void Evaluate(System.Collections.Generic.IDictionary<Variable, Value> inputs, System.Collections.Generic.IDictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
    {
        Evaluate(inputs, outputs, false, computeDevice);
    }

    // Evaluates the Function using provided inputs.
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

        _Evaluate(inMap, outMap, computeDevice);

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

    // Find the function with the specified name.
    public Function FindByName(string name, bool nestedSearchInsideBlockFunction = false)
    {
        return _FindByName(name, nestedSearchInsideBlockFunction);
    }

    // Finds all functions inside this Functions having the specified name.
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

    // Loads a model from file.
    public static Function Load(string filepath, DeviceDescriptor computeDevice)
    {
        return _Load(filepath, computeDevice);
    }

    // Loads a model from memory buffer.
    public static Function Load(byte[] modelBuffer, DeviceDescriptor computeDevice)
    {
        return _Load(modelBuffer, (uint)modelBuffer.Length, computeDevice);
    }

    // Creates a new Function from specified operands.
    public static Function Combine(System.Collections.Generic.IEnumerable<Variable> operands)
    {
        var varVect = new VariableVector();
        foreach (var v in operands)
        {
            varVect.Add(v);
        }
        return CNTKLib.Combine(varVect);
    }

    // Creates a composite function from the rootFunction.
    public static Function AsComposite(Function rootFunction, string name = "")
    {
        return CNTKLib.AsComposite(rootFunction, name);
    }

    // Create a new Function which is the alias of operand.
    public static Function Alias(Variable operand, string name = "")
    {
        return CNTKLib.Alias(operand, name);
    }
%}

%typemap(cscode) CNTK::Variable %{

    // Property Shape.
    public NDShape Shape
    {
        get { return _Shape(); }
    }

    // Property Name.
    public string Name
    {
        get { return _Name(); }
    }

    // Property Uid.
    public string Uid
    {
        get { return _Uid(); }
    }

    // Property Kind.
    public VariableKind Kind
    {
        get { return _Kind(); }
    }

    // Property DataType.
    public DataType DataType
    {
        get { return _GetDataType(); }
    }

    // Property DynamicAxes.
    public System.Collections.Generic.IList<Axis> DynamicAxes
    {
        get {
            var axisVector = _DynamicAxes();
            // The CopyTo is to ensure that elements in axisVector live beyond the lifecycle of axisVector.
            var axisArray = new Axis[axisVector.Count];
            axisVector.CopyTo(axisArray);
            var axisList = new System.Collections.Generic.List<Axis>(axisArray);
            return axisList;
        }
    }

    // Property IsSparse.
    public bool IsSparse
    {
        get { return _IsSparse(); }
    }

    // Property IsInput.
    public bool IsInput
    {
        get { return _IsInput(); }
    }

    // Property IsOutput.
    public bool IsOutput
    {
        get { return _IsOutput(); }
    }

    // Property IsParameter.
    public bool IsParameter
    {
        get { return _IsParameter(); }
    }

    // Property IsConstant.
    public bool IsConstant
    {
        get { return _IsConstant(); }
    }

    // Property IsPlaceholder.
    public bool IsPlaceholder
    {
        get { return _IsPlaceholder(); }
    }

    // Property Owner.
    public Function Owner
    {
        get { return _Owner(); }
    }

    // Property NeedsGradient.
    public bool NeedsGradient
    {
        get { return _NeedsGradient(); }
    }

    // Property CurrentValueTimeStamp
    public int CurrentValueTimeStamp
    {
        get { return (int)_CurrentValueTimeStamp(); }
    }

    // Value equality.
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
        return CNTKLib.AreEqual(this, p);
    }

    // Value equality.
    public bool Equals(Variable p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Returns hash code value.
    public override int GetHashCode()
    {
        // Todo: the hash value in C++ is size_t, but only in C#
        return (int)_GetHashValue();
    }
%}

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

    // Property Rank.
    public int Rank
    {
        get { return (int)_Rank(); }
    }

    // Property Dimensions.
    public System.Collections.Generic.IList<int> Dimensions
    {
        get
        {
            var dimList = _Dimensions();
            var retList = new System.Collections.Generic.List<int>(dimList.Count);
            foreach (var element in dimList)
            {
                retList.Add((int)element);
            }
            return retList;
        }
    }

    // Property IsUnknown.
    public bool IsUnknown 
    {
        get { return _IsUnknown(); }
    }

    // Property HasInferredDimension.
    public bool HasInferredDimension
    {
        get { return _HasInferredDimension(); }
    }

    // Property HasFreeDimension.
    public bool HasFreeDimension
    {
        get { return _HasFreeDimension(); }
    }

    // Property HasUnboundDimension.
    public bool HasUnboundDimension
    {
        get { return _HasUnboundDimension(); }
    }

    // Property TotalSize.
    public int TotalSize
    {
        get { return (int)_TotalSize(); }
    }

    // Indexer operator
    public int this[int key]
    {
        get { return (int)_DimensionSize((uint)key); }
    }

    // Returns a subshape.
    public NDShape SubShape(int beginAxisId, int endAxisId)
    {
        if (beginAxisId < 0 || endAxisId < 0)
        {
            throw new System.ArgumentException("The paraemter beginAxisId or endAxisId should not be a negative value");
        }
        return _SubShape((uint)beginAxisId, (uint)endAxisId);
    }

    // Returns a subshape.
    public NDShape SubShape(int beginAxisId = 0)
    {
        if (beginAxisId < 0)
        {
            throw new System.ArgumentException("The paraemter beginAxisId should not be a negative value");
        }
        return _SubShape((uint)beginAxisId);
    }

    // Creates a new NDShape.
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

    // Value equality.
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
        return CNTKLib.AreEqual(this, p);
    }

    // Value Equality.
    public bool Equals(NDShape p)
    {
        // If parameter is null return false:
        if ((object)p == null)
        {
            return false;
        }

        // Return true if the fields match:
        return CNTKLib.AreEqual(this, p);
    }

    // Returns hash code value.
    public override int GetHashCode()
    {
        //Todo: another hash function??
        return this._Dimensions().GetHashCode();
    }

    // Constants
    public static readonly int InferredDimension = -1;
    public static readonly int FreeDimension = -3;
%}

%typemap(cscode) CNTK::NDMask %{

    // Property MaskedCount.
    public int MaskedCount {
        get { return (int)_MaskedCount(); }
    }

    // Property Device.
    public DeviceDescriptor Device {
        get { return _Device(); }
    }

    // Property Shape.
    public NDShape Shape {
        get { return _Shape(); }
    }

    // Invidates a section of a NDShape.
    public void InvalidateSection(System.Collections.Generic.IEnumerable<int> sectionOffset, NDShape sectionShape) {
        var offsetVector = Helper.AsSizeTVector(sectionOffset);
        _InvalidateSection(offsetVector, sectionShape);
    }

    // Marks sequence begin.
    public void MarkSequenceBegin(System.Collections.Generic.IEnumerable<int> offset) {
        var offsetVector = Helper.AsSizeTVector(offset);
        _MarkSequenceBegin(offsetVector);
    }

    // Marks sequence begins in a NDShape.
    public void MarkSequenceBegin(System.Collections.Generic.IEnumerable<int> offset, NDShape sectionShape) {
        var offsetVector = Helper.AsSizeTVector(offset);
        _MarkSequenceBegin(offsetVector, sectionShape);
    }
%}

%typemap(cscode) CNTK::Value %{

    // Property Device
    public DeviceDescriptor Device
    {
        get { return _Device(); }
    }

    // Property DataType
    public DataType DataType
    {
        get { return _GetDataType(); }
    }

    // Property StorageFormat
    public StorageFormat StorgeFormat
    {
        get { return _GetStorageFormat(); }
    }

    // Property Shape
    public NDShape Shape
    {
        get { return _Shape(); }
    }

    // Property IsValid
    public bool IsValid
    {
        get { return _IsValid(); }
    }

    // Property IsSparse
    public bool IsSparse
    {
        get { return _IsSparse(); }
    }

    // Property IsReadOnly
    public bool IsReadOnly
    {
        get { return _IsReadOnly(); }
    }

    // Property MaskedCount
    public int MaskedCount
    {
        get { return (int)_MaskedCount(); }
    }

    // Property Data
    public NDArrayView Data
    {
        get { return _Data(); }
    }

    // Property Mask
    public NDMask Mask
    {
        get { return _Mask(); }
    }

    // Create Value object from dense input as batch data.
    public static Value CreateBatch<T>(NDShape sampleShape, System.Collections.Generic.IEnumerable<T> batch, DeviceDescriptor device, bool readOnly = false)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            var inputVector = Helper.AsFloatVector(batch);
            return Value._CreateBatchFloat(sampleShape, inputVector, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputVector = Helper.AsDoubleVector(batch);
            return Value._CreateBatchDouble(sampleShape, inputVector, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from dense input as sequence data.
    public static Value CreateSequence<T>(NDShape sampleShape,
                                          System.Collections.Generic.IEnumerable<T> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(sampleShape, sequence, true, device, readOnly);
    }

    // Create Value object from dense input as sequence data with sequenceStartFlag.
    public static Value CreateSequence<T>(NDShape sampleShape,
                                          System.Collections.Generic.IEnumerable<T> sequence,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        if (typeof(T).Equals(typeof(float)))
        {
            var inputVector = Helper.AsFloatVector(sequence);
            return Value._CreateSequenceFloat(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputVector = Helper.AsDoubleVector(sequence);
            return Value._CreateSequenceDouble(sampleShape, inputVector, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from dense input as batch of sequences data.
    public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<T>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(sampleShape, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<T>> batchOfSequences,
                                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    // Create Value object from dense input as batch of sequences data with sequenceStartFlags.
    public static Value Create<T>(NDShape sampleShape,
                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<T>> sequences,
                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        if (typeof(T).Equals(typeof(float)))
        {
            var inputAsSequencesVector = new FloatVectorVector();
            foreach (var seq in sequences)
            {
                var seqVector = Helper.AsFloatVector(seq);
                // The seqVector is copied when adding to inputAsSequencesVector.
                inputAsSequencesVector.Add(seqVector);
            }
            return Value._CreateDenseFloat(sampleShape, inputAsSequencesVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            var inputAsSequencesVector = new DoubleVectorVector();
            foreach (var seq in sequences)
            {
                var seqVector = Helper.AsDoubleVector(seq);
                inputAsSequencesVector.Add(seqVector);
            }
            return Value._CreateDenseDouble(sampleShape, inputAsSequencesVector, seqFlags, device, readOnly);
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
        var seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        var inputSeqVector = new SizeTVectorVector();
        foreach (var seq in sequences)
        {
            var s = Helper.AsSizeTVector(seq);
            inputSeqVector.Add(s);
        }
        if (typeof(T).Equals(typeof(float)))
        {
            return Value._CreateOneHotFloat(sampleShape, inputSeqVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value._CreateOneHotDouble(sampleShape, inputSeqVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input as batch data, for 1D tensor only.
    public static Value CreateBatch<T>(int dimension, System.Collections.Generic.IEnumerable<int> batch, DeviceDescriptor device, bool readOnly = false)
    {
        var inputVector = Helper.AsSizeTVector(batch);
        if (typeof(T).Equals(typeof(float)))
        {
            return Value._CreateBatchFloat((uint)dimension, inputVector, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value._CreateBatchDouble((uint)dimension, inputVector, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input as sequence data, for 1D tensor only.
    public static Value CreateSequence<T>(int dimension,
                                          System.Collections.Generic.IEnumerable<int> sequence,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return CreateSequence<T>(dimension, sequence, true, device, readOnly);
    }

    // Create Value object from OneHotVector input as sequence data with sequenceStartFlag, for 1D tensor only.
    public static Value CreateSequence<T>(int dimension,
                                          System.Collections.Generic.IEnumerable<int> sequence,
                                          bool sequenceStartFlag,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        var inputVector = Helper.AsSizeTVector(sequence);
        if (typeof(T).Equals(typeof(float)))
        {
            return Value._CreateSequenceFloat((uint)dimension, inputVector, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value._CreateSequenceDouble((uint)dimension, inputVector, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from OneHotVector input as batch of sequences data, for 1D tensor only.
    public static Value CreateBatchOfSequences<T>(int dimension,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> batchOfSequences,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    // Create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
    public static Value CreateBatchOfSequences<T>(int dimension,
                                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> batchOfSequences,
                                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                                  DeviceDescriptor device,
                                                  bool readOnly = false)
    {
        return Create<T>(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
    }

    // Create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
    public static Value Create<T>(int dimension,
                                  System.Collections.Generic.IEnumerable<System.Collections.Generic.IEnumerable<int>> sequences,
                                  System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                                  DeviceDescriptor device,
                                  bool readOnly = false)
    {
        var seqFlags = Helper.AsBoolVector(sequenceStartFlags);
        var inputSeqVector = new SizeTVectorVector();
        foreach (var seq in sequences)
        {
            var s = Helper.AsSizeTVector(seq);
            inputSeqVector.Add(s);
        }
        if (typeof(T).Equals(typeof(float)))
        {
            return Value._CreateOneHotFloat((uint)dimension, inputSeqVector, seqFlags, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value._CreateOneHotDouble((uint)dimension, inputSeqVector, seqFlags, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from sparse input as sequence data with sequenceStartFlag, for N-dimensional tensor. Only CreateSequence() for now.
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
            return Value._CreateSequenceFloat(sampleShape, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as float[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value._CreateSequenceDouble(sampleShape, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as double[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from sparse input as sequence data, for N-dimensional tensor. Only CreateSequence() for now.
    public static Value CreateSequence<T>(NDShape sampleShape, int sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return Value.CreateSequence<T>(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, true, device, readOnly);
    }

    // Create Value object from sparse input as sequence data with sequenceStartFlag, for 1D tensor. Only CreateSequence() for now.
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
            return Value._CreateSequenceFloat((uint)dimension, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as float[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            return Value._CreateSequenceDouble((uint)dimension, (uint)sequenceLength, colStarts, rowIndices, nonZeroValues as double[], numNonZeroValues, sequenceStartFlag, device, readOnly);
        }
        else
        {
            throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
        }
    }

    // Create Value object from sparse input as sequence data, for 1D tensor. Only CreateSequence() for now.
    public static Value CreateSequence<T>(int dimension, int sequenceLength,
                                          int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                          DeviceDescriptor device,
                                          bool readOnly = false)
    {
        return Value.CreateSequence<T>(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, true, device, readOnly);
    }

    // Create Value object from NDArrayViews.
    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.IEnumerable<NDArrayView> sequences,
                               DeviceDescriptor device,
                               bool readOnly = false)
    {
        return Create(sampleShape, sequences, new System.Collections.Generic.List<bool>(0), device, readOnly);
    }

    // Create Value object from NDArrayViews with sequenceStartFlags
    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.IEnumerable<NDArrayView> sequences,
                               System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                               DeviceDescriptor device,
                               bool readOnly = false)
    {
        return Create(sampleShape, sequences, sequenceStartFlags, device, readOnly, /*createNewCopy = */ false);
    }

    // Create Value object from NDArrayViews with sequenceStartFlags
    public static Value Create(NDShape sampleShape,
                               System.Collections.Generic.IEnumerable<NDArrayView> sequences,
                               System.Collections.Generic.IEnumerable<bool> sequenceStartFlags,
                               DeviceDescriptor device,
                               bool readOnly,
                               bool createNewCopy)
    {
        var seqVector = new NDArrayViewPtrVector();
        foreach (var element in sequences)
        {
            seqVector.Add(element);
        }
        var startFlags = Helper.AsBoolVector(sequenceStartFlags);
        return _Create(sampleShape, seqVector, startFlags, device, readOnly, createNewCopy);
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
            if (_GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new FloatVectorVector();
            _CopyVariableValueToFloat(outputVariable, seqVec);

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
            if (_GetDataType() != DataType.Double)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new DoubleVectorVector();
            _CopyVariableValueToDouble(outputVariable, seqVec);
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
        _CopyVariableValueTo(outputVariable, seqVec);
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
            if (_GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new FloatVectorVector();
            _CopyVariableValueToFloat(outputVariable, seqVec);

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
            if (_GetDataType() != DataType.Double)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var seqVec = new DoubleVectorVector();
            _CopyVariableValueToDouble(outputVariable, seqVec);
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
        _CopyVariableValueTo(outputVariable, seqVec);

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

    public void GetSparseData<T>( Variable outputVariable,
                                    out int sequenceLength,
                                    out System.Collections.Generic.IList<int> colStarts,
                                    out System.Collections.Generic.IList<int> rowIndices,
                                    out System.Collections.Generic.IList<T> nonZeroValues,
                                    out int numNonZeroValues) 
    {
        var colStartVec = new IntVector();
        var rowIndicesVec = new IntVector();

        int[] n1 = new int[1];
        int[] n2 = new int[1];

        if (typeof(T).Equals(typeof(float)))
        {
            if (_GetDataType() != DataType.Float)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var nonZeroValuesVec = new FloatVector();
            _CopyVariableValueToFloat(outputVariable, n1,  colStartVec,   
                rowIndicesVec, nonZeroValuesVec, n2);
            nonZeroValues = nonZeroValuesVec as System.Collections.Generic.IList<T>;
        }
        else if (typeof(T).Equals(typeof(double)))
        {
            if (_GetDataType() != DataType.Double)
            {
                throw new System.ArgumentException("The value type does not match the list type.");
            }

            var nonZeroValuesVec = new DoubleVector();
            _CopyVariableValueToDouble(outputVariable, n1, colStartVec,
                rowIndicesVec, nonZeroValuesVec, n2);
            nonZeroValues = nonZeroValuesVec as System.Collections.Generic.IList<T>;
        }
        else
        {
            throw new System.ArgumentException("The value type does not match the list type.");
        }

        sequenceLength = n1[0];
        numNonZeroValues = n2[0];
        colStarts = colStartVec;
        rowIndices = rowIndicesVec;
    }


    // Creates a new Value which is an alias of this Value.
    public Value Alias(bool readOnly = false)
    {
        return _Alias(readOnly);
    }

%}

%typemap(cscode) CNTK::NDArrayView %{

    // Constructor using float dense input.
    public NDArrayView(NDShape viewShape, float[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
    {
    }

    // Constructor using double dense input.
    public NDArrayView(NDShape viewShape, double[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
    {
    }

    // Constructor using float sparse input.
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

    // Constructor using double sparse input.
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

    // Property Device.
    public DeviceDescriptor Device
    {
        get { return _Device(); }
    }

    // Property DataType.
    public DataType DataType
    {
        get { return _GetDataType(); }
    }

    // Property Shape.
    public NDShape Shape
    {
        get { return _Shape(); }
    }

    // Property StorageFormat.
    public StorageFormat StorageFormat
    {
        get { return _GetStorageFormat(); }
    }

    // Property IsSparse.
    public bool IsSparse
    {
        get { return _IsSparse(); }
    }

    // Property IsReadOnly.
    public bool IsReadOnly
    {
        get { return _IsReadOnly(); }
    }

    // Returns a slice view.
    public NDArrayView SliceView(System.Collections.Generic.IEnumerable<int> startOffset, System.Collections.Generic.IEnumerable<int> extent, bool readOnly = false)
    {
        var startOffsetVector = Helper.AsSizeTVector(startOffset);

        var extentVector = Helper.AsSizeTVector(extent);

        return _SliceView(startOffsetVector, extentVector, readOnly);
    }

    // Creates a new NDArrayView which is an alias of this NDArrayView.
    public NDArrayView Alias(bool readOnly = false)
    {
        return _Alias(readOnly);
    }
%}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
