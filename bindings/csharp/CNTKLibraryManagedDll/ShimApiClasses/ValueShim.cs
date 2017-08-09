//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ValueShim.cs -- C# Api for CNTK Value class
//
namespace CNTK
{
    public partial class Value
    {
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
            foreach (var seq in seqVec)
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
            foreach (var seq in seqVec)
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

        public void GetSparseData<T>(Variable outputVariable,
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
                _CopyVariableValueToFloat(outputVariable, n1, colStartVec,
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

    }
}
