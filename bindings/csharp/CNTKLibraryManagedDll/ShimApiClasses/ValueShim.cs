//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ValueShim.cs -- C# Api for CNTK Value class
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class Value
    {
        /// <summary>
        /// Property Device
        /// </summary>
        public DeviceDescriptor Device
        {
            get { return _Device(); }
        }

        /// <summary>
        /// Property DataType
        /// </summary>
        public DataType DataType
        {
            get { return _GetDataType(); }
        }

        /// <summary>
        /// Property StorageFormat
        /// </summary>
        public StorageFormat StorgeFormat
        {
            get { return _GetStorageFormat(); }
        }

        /// <summary>
        /// Property Shape
        /// </summary>
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        /// <summary>
        /// Property IsValid
        /// </summary>
        public bool IsValid
        {
            get { return _IsValid(); }
        }

        /// <summary>
        /// Property IsSparse
        /// </summary>
        public bool IsSparse
        {
            get { return _IsSparse(); }
        }

        /// <summary>
        /// Property IsReadOnly
        /// </summary>
        public bool IsReadOnly
        {
            get { return _IsReadOnly(); }
        }

        /// <summary>
        /// Property MaskedCount
        /// </summary>
        public int MaskedCount
        {
            get { return (int)_MaskedCount(); }
        }

        /// <summary>
        /// Property Data
        /// </summary>
        public NDArrayView Data
        {
            get { return _Data(); }
        }

        /// <summary>
        /// Property Mask
        /// </summary>
        public NDMask Mask
        {
            get { return _Mask(); }
        }

        /// <summary>
        /// Create Value object from dense input as batch data.
        /// </summary>
        /// <typeparam name="T">float or double</typeparam>
        /// <param name="sampleShape">shape of the Value</param>
        /// <param name="batch">batch of data</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">readonly value</param>
        /// <returns>the value</returns>
        public static Value CreateBatch<T>(NDShape sampleShape, IEnumerable<T> batch, DeviceDescriptor device, bool readOnly = false)
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from dense input as sequence data.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sampleShape">the shape fo the value</param>
        /// <param name="sequence">daat sequence</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it is a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(NDShape sampleShape,
                                              IEnumerable<T> sequence,
                                              DeviceDescriptor device,
                                              bool readOnly = false)
        {
            return CreateSequence<T>(sampleShape, sequence, true, device, readOnly);
        }

        /// <summary>
        /// Create Value object from dense input as sequence data with sequenceStartFlag.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="sampleShape">data shape</param>
        /// <param name="sequence">data sequence</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence. false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(NDShape sampleShape,
                                              IEnumerable<T> sequence,
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from dense input as batch of sequences data.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="sampleShape">data shape</param>
        /// <param name="batchOfSequences">the data to be stored in the Value.
        /// The outer vector represents a collection of sequences with variable length, 
        /// and the inner vector represents each individual sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                      IEnumerable<IEnumerable<T>> batchOfSequences,
                                                      DeviceDescriptor device,
                                                      bool readOnly = false)
        {
            return Create(sampleShape, batchOfSequences, new List<bool>(0), device, readOnly);
        }

        /// <summary>
        /// Create Value object from dense input as batch of sequences data with sequenceStartFlags.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="sampleShape">data shape</param>
        /// <param name="batchOfSequences">the data to be stored in the Value.
        /// The outer vector represents a collection of sequences with variable length, 
        /// and the inner vector represents each individual sequence.</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateBatchOfSequences<T>(NDShape sampleShape,
                                                      IEnumerable<IEnumerable<T>> batchOfSequences,
                                                      IEnumerable<bool> sequenceStartFlags,
                                                      DeviceDescriptor device,
                                                      bool readOnly = false)
        {
            return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
        }

        /// <summary>
        /// Create Value object from dense input as batch of sequences data with sequenceStartFlags.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="sampleShape">data shape</param>
        /// <param name="sequences">data sequence</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value Create<T>(NDShape sampleShape,
                                      IEnumerable<IEnumerable<T>> sequences,
                                      IEnumerable<bool> sequenceStartFlags,
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from OneHotVector input, for N-dimenstional tensor. Only Create() method for now.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="sampleShape">data shape</param>
        /// <param name="sequences">data sequence</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value Create<T>(NDShape sampleShape,
                                      IEnumerable<IEnumerable<int>> sequences,
                                      IEnumerable<bool> sequenceStartFlags,
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from OneHotVector input as batch data, for 1D tensor only.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="batch">data batches</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateBatch<T>(int dimension, IEnumerable<int> batch, DeviceDescriptor device, bool readOnly = false)
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from OneHotVector input as sequence data, for 1D tensor only.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="sequence">data sequence</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(int dimension,
                                              IEnumerable<int> sequence,
                                              DeviceDescriptor device,
                                              bool readOnly = false)
        {
            return CreateSequence<T>(dimension, sequence, true, device, readOnly);
        }

        /// <summary>
        /// Create Value object from OneHotVector input as sequence data with sequenceStartFlag, for 1D tensor only.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="sequence">data sequence</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(int dimension,
                                              IEnumerable<int> sequence,
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from OneHotVector input as batch of sequences data, for 1D tensor only.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="batchOfSequences">data sequence batches</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateBatchOfSequences<T>(int dimension,
                                                      IEnumerable<IEnumerable<int>> batchOfSequences,
                                                      DeviceDescriptor device,
                                                      bool readOnly = false)
        {
            return Create<T>(dimension, batchOfSequences, new List<bool>(0), device, readOnly);
        }

        /// <summary>
        /// Create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="batchOfSequences">data sequence batches</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateBatchOfSequences<T>(int dimension,
                                                      IEnumerable<IEnumerable<int>> batchOfSequences,
                                                      IEnumerable<bool> sequenceStartFlags,
                                                      DeviceDescriptor device,
                                                      bool readOnly = false)
        {
            return Create<T>(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
        }

        /// <summary>
        /// Create Value object from OneHotVector input as batch of sequences data with sequenceStratFlags, for 1D tensor only.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="sequences">data sequences</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value Create<T>(int dimension,
                                      IEnumerable<IEnumerable<int>> sequences,
                                      IEnumerable<bool> sequenceStartFlags,
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from sparse input as sequence data with sequenceStartFlag, for N-dimensional tensor. Only CreateSequence() for now.
        /// </summary>
        /// <typeparam name="T">data type of the created Value object.Currently, float and double are supported.</typeparam>
        /// <param name="sampleShape">the tensor shape. For sparse input, the tensor shape leading dimensionality must be the same as the total size of the tensor shape.</param>
        /// <param name="sequenceLength">the sequence length.</param>
        /// <param name="colStarts">column start indices</param>
        /// <param name="rowIndices">row indices</param>
        /// <param name="nonZeroValues">sparse values</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(NDShape sampleShape, int sequenceLength,
                                              int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                              bool sequenceStartFlag,
                                              DeviceDescriptor device,
                                              bool readOnly = false)
        {
            if (nonZeroValues.Length != rowIndices.Length)
            {
                throw new ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
            }
            if (colStarts.Length != sequenceLength + 1)
            {
                throw new ArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from sparse input as sequence data, for N-dimensional tensor. Only CreateSequence() for now.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="sampleShape">the tensor shape. For sparse input, the tensor shape leading dimensionality must be the same as the total size of the tensor shape.</param>
        /// <param name="sequenceLength">the sequence length.</param>
        /// <param name="colStarts">column start indices</param>
        /// <param name="rowIndices">row indices</param>
        /// <param name="nonZeroValues">sparse values</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(NDShape sampleShape, int sequenceLength,
                                              int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                              DeviceDescriptor device,
                                              bool readOnly = false)
        {
            return Value.CreateSequence<T>(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, true, device, readOnly);
        }

        /// <summary>
        /// Create Value object from sparse input as sequence data with sequenceStartFlag, for 1D tensor. Only CreateSequence() for now.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dimension"></param>
        /// <param name="sequenceLength">the sequence length.</param>
        /// <param name="colStarts">column start indices</param>
        /// <param name="rowIndices">row indices</param>
        /// <param name="nonZeroValues">sparse values</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(int dimension, int sequenceLength,
                                              int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                              bool sequenceStartFlag,
                                              DeviceDescriptor device,
                                              bool readOnly = false)
        {
            if (nonZeroValues.Length != rowIndices.Length)
            {
                throw new ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
            }
            if (colStarts.Length != sequenceLength + 1)
            {
                throw new ArgumentException("The length of colStarts must be equal to (sequenceLength + 1)");
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Create Value object from sparse input as sequence data, for 1D tensor. Only CreateSequence() for now.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="dimension">data dimension</param>
        /// <param name="sequenceLength">the sequence length.</param>
        /// <param name="colStarts">column start indices</param>
        /// <param name="rowIndices">row indices</param>
        /// <param name="nonZeroValues">sparse values</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value CreateSequence<T>(int dimension, int sequenceLength,
                                              int[] colStarts, int[] rowIndices, T[] nonZeroValues,
                                              DeviceDescriptor device,
                                              bool readOnly = false)
        {
            return Value.CreateSequence<T>(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, true, device, readOnly);
        }

        /// <summary>
        /// Create Value object from NDArrayViews.
        /// </summary>
        /// <param name="sampleShape">data shape</param>
        /// <param name="sequences">data sequence</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value Create(NDShape sampleShape,
                                   IEnumerable<NDArrayView> sequences,
                                   DeviceDescriptor device,
                                   bool readOnly = false)
        {
            return Create(sampleShape, sequences, new List<bool>(0), device, readOnly);
        }

        /// <summary>
        /// Create Value object from NDArrayViews with sequenceStartFlags
        /// </summary>
        /// <param name="sampleShape">data shape</param>
        /// <param name="sequences">data sequences</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <returns></returns>
        public static Value Create(NDShape sampleShape,
                                   IEnumerable<NDArrayView> sequences,
                                   IEnumerable<bool> sequenceStartFlags,
                                   DeviceDescriptor device,
                                   bool readOnly = false)
        {
            return Create(sampleShape, sequences, sequenceStartFlags, device, readOnly, /*createNewCopy = */ false);
        }

        /// <summary>
        /// Create Value object from NDArrayViews with sequenceStartFlags
        /// </summary>
        /// <param name="sampleShape">data shape</param>
        /// <param name="sequences">data sequence</param>
        /// <param name="sequenceStartFlag">true indicates that it is a new sequence.
        /// false means a continuation of a previous sequence.</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether it a readonly value</param>
        /// <param name="createNewCopy"></param>
        /// <returns></returns>
        public static Value Create(NDShape sampleShape,
                                   IEnumerable<NDArrayView> sequences,
                                   IEnumerable<bool> sequenceStartFlags,
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

        /// <summary>
        /// Return the data of the Value object as a list of sequences with variable length.
        /// This method returns an IList<IList<T>>. Each element of the outer list represents a sequence.
        /// Each sequence, represented by IList<T>, contains a variable number of samples.
        /// Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the variable shape.
        /// The number of samples = (the count of elements in IList<T>)/(the count of elements of the sample)
        /// The shape of the variable should match the shape of the Value object.
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="outputVariable">the source variable</param>
        /// <returns></returns>
        public IList<IList<T>> GetDenseData<T>(Variable outputVariable)
        {
            var sequences = new List<IList<T>>();
            if (typeof(T).Equals(typeof(float)))
            {
                if (_GetDataType() != DataType.Float)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var seqVec = new FloatVectorVector();
                _CopyVariableValueToFloat(outputVariable, seqVec);

                foreach (var seq in seqVec)
                {
                    var seqList = seq as IList<T>;
                    if (seqList == null)
                        throw new TypeAccessException("Cannot convert to the value type.");
                    // It is required to create a new List from seq, since seq is dependent on the life cycle of seqVec.
                    sequences.Add(new List<T>(seqList));
                }
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                if (_GetDataType() != DataType.Double)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var seqVec = new DoubleVectorVector();
                _CopyVariableValueToDouble(outputVariable, seqVec);
                foreach (var seq in seqVec)
                {
                    var seqList = seq as IList<T>;
                    if (seqList == null)
                        throw new TypeAccessException("Cannot convert to the value type.");
                    // It is required to create a new List from seq, since seq is dependent on the life cycle of seqVec.
                    sequences.Add(new List<T>(seqList));
                }
            }
            else
            {
                throw new ArgumentException("The value type does not match the list type.");
            }
            return sequences;
        }

        /// <summary>
        /// Return the data of the Value object as a list of sequences with variable length.
        /// This method returns an IList<IList<T>>. Each element of the outer list represents a sequence.
        /// Each sequence, represented by List<int>, contains a variable number of samples.
        /// Each sample is represented by an index of the OneHot vector. The size of the OneHot vector should match that defined in the variable.
        /// The number of samples = the count of elements in List<int>.
        /// </summary>
        /// <param name="outputVariable">the source variable</param>
        /// <returns></returns>
        public IList<IList<int>> GetOneHotData(Variable outputVariable)
        {
            var sequences = new List<IList<int>>();
            var seqVec = new SizeTVectorVector();
            _CopyVariableValueTo(outputVariable, seqVec);
            foreach (var seq in seqVec)
            {
                var seqList = new List<int>(seq.Count);
                foreach (var element in seq)
                {
                    seqList.Add((int)element);
                }
                sequences.Add(seqList);
            }
            return sequences;
        }

        /// <summary>
        /// Copy the data of the Value object into the buffer provided by 'sequences'.
        /// The 'sequences' is a list of sequences with variable length. 
        /// The number of items contained in the outer list of 'sequences' is the number of sequences in the Value object.
        /// Each element of the outer list represents a sequence.
        /// Each sequence, represented by List<T>, contains a variable number of samples. 
        /// Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the variable shape.
        /// The number of samples = the count of elements in List<T> / the count of elements of the sample
        /// The shape of the variable should match the shape of the Value object.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="outputVariable"></param>
        /// <param name="sequences"></param>
        [Obsolete("CopyVariableValueTo() will be deprecated soon. Please use GetDenseData() instead.")]
        public void CopyVariableValueTo<T>(Variable outputVariable, List<List<T>> sequences)
        {
            sequences.Clear();
            if (typeof(T).Equals(typeof(float)))
            {
                if (_GetDataType() != DataType.Float)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var seqVec = new FloatVectorVector();
                _CopyVariableValueToFloat(outputVariable, seqVec);

                foreach (var seq in seqVec)
                {
                    var seqList = seq as IList<T>;
                    if (seqList == null)
                        throw new TypeAccessException("Cannot convert to the value type.");
                    sequences.Add(new List<T>(seqList));
                }
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                if (_GetDataType() != DataType.Double)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var seqVec = new DoubleVectorVector();
                _CopyVariableValueToDouble(outputVariable, seqVec);
                foreach (var seq in seqVec)
                {
                    var seqList = seq as IList<T>;
                    if (seqList == null)
                        throw new TypeAccessException("Cannot convert to the value type.");
                    sequences.Add(new List<T>(seqList));
                }
            }
            else
            {
                throw new ArgumentException("The value type does not match the list type.");
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
        [Obsolete("CopyVariableValueTo() will be deprecated soon. Please use GetOneHotData() instead.")]
        public void CopyVariableValueTo(Variable outputVariable, List<List<int>> sequences)
        {
            var seqVec = new SizeTVectorVector();
            _CopyVariableValueTo(outputVariable, seqVec);

            sequences.Clear();
            foreach (var seq in seqVec)
            {
                var seqList = new List<int>(seq.Count);
                foreach (var element in seq)
                {
                    seqList.Add((int)element);
                }
                sequences.Add(seqList);
            }
            return;
        }

        /// <summary>
        /// Copy the data stored in 'this' Value object to the buffers representing a sequence in CSC sparse format.
        /// The sequence buffer will be resized if necessary.
        /// The Value should have the same tensor shape as outputVariable.
        /// On return, the sequenceLength is set to the length of the sequence stored in 'this' Value,
        /// and the colStarts, rowIndices and nonZeroValues contain the data of column indexes, row indexes and non-zero values,
        /// and the numNonZeroValues is set to number of non-zero values contained in 'this' Value.
        /// </summary>
        /// <typeparam name="T">dat type</typeparam>
        /// <param name="outputVariable">source variable</param>
        /// <param name="sequenceLength">legnth of the sequence</param>
        /// <param name="colStarts">column start indices</param>
        /// <param name="rowIndices">row indices</param>
        /// <param name="nonZeroValues">sparse values</param>
        /// <param name="numNonZeroValues">number of sparse values</param>
        public void GetSparseData<T>(Variable outputVariable,
                                        out int sequenceLength,
                                        out IList<int> colStarts,
                                        out IList<int> rowIndices,
                                        out IList<T> nonZeroValues,
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
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var nonZeroValuesVec = new FloatVector();
                _CopyVariableValueToFloat(outputVariable, n1, colStartVec,
                    rowIndicesVec, nonZeroValuesVec, n2);
                nonZeroValues = nonZeroValuesVec as IList<T>;
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                if (_GetDataType() != DataType.Double)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var nonZeroValuesVec = new DoubleVector();
                _CopyVariableValueToDouble(outputVariable, n1, colStartVec,
                    rowIndicesVec, nonZeroValuesVec, n2);
                nonZeroValues = nonZeroValuesVec as IList<T>;
            }
            else
            {
                throw new ArgumentException("The value type does not match the list type.");
            }

            sequenceLength = n1[0];
            numNonZeroValues = n2[0];
            colStarts = colStartVec;
            rowIndices = rowIndicesVec;
        }

        /// <summary>
        /// Creates a new Value which is an alias of this Value.
        /// </summary>
        /// <param name="readOnly"></param>
        /// <returns></returns>
        public Value Alias(bool readOnly = false)
        {
            return _Alias(readOnly);
        }

    }
}
