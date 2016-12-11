//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ValueExtendsions.cs -- Define extension methods for Value.
//
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace CNTK
{
    public static class ValueExtensions
    {

        // The value represents a n-dimensional tensor with 2 dynamic axes: sequence and batch
        // It assumes that only the highest 2 axes are dynamic, and all the other axes are static. 
        public static void CopyTo<T>(this Value value, Variable variable, List<List<T>> data)
        {
            if ((value.GetDataType() == DataType.Float) && (!typeof(T).Equals(typeof(float))) || 
                (value.GetDataType() == DataType.Double) && (!typeof(T).Equals(typeof(double))))
            {
                throw new ArgumentException("The value type does not match the list type.");
            }

            // Todo: how to check whether the dynamic axes are the highest 2 axes in the shape.
            if (variable.DynamicAxes().Count != 2)
            {
                throw new ArgumentException("The variable should have 2 dynamic axes.");
            }

            var variableShape = variable.Shape;
            var valueShape = value.Shape;

            // Todo: can a value only have the sequencee axis, but no batch axis??
            if (variableShape != value.Shape.SubShape(0, valueShape.Rank - 2))
            {
                throw new ArgumentException("The variable and value does not have same shape.");
            }

            // Todo: transform sparse to dense
            // Currently only for dense
            if ((value.GetStorageFormat() != StorageFormat.Dense))
            {
                throw new ArgumentException("The value is not in denst format.");
            }

            var outputNDArrayView = value.Data();
            var outputShape = outputNDArrayView.Shape();
            var outputShapeRank = outputShape.Rank;
            var numOfElementsInSample = variableShape.TotalSize;
            var numOfSamplesInSequence = outputShape.GetDimensionSize(outputShapeRank - 2);
            var numOfSequences = outputShape.GetDimensionSize(outputShapeRank - 1);

            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer?
            // Todo: need to map DataBuffer() to C#
            NDArrayView cpuOutputNDArrayView;
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize;
            // Todo: consider mask.
            Debug.Assert(numOfElementsInSample * numOfSamplesInSequence * numOfSequences == numOfOutputData);
            T[] outputData = new T[numOfOutputData];
            if (value.GetDataType() == DataType.Float)
            {
                cpuOutputNDArrayView = new NDArrayView(outputNDArrayView.Shape(), outputData as float[], numOfOutputData, DeviceDescriptor.CPUDevice);
            }
            else if (value.GetDataType() == DataType.Double)
            {
                cpuOutputNDArrayView = new NDArrayView(outputNDArrayView.Shape(), outputData as double[], numOfOutputData, DeviceDescriptor.CPUDevice);
            }
            else
            {
                throw new ArgumentException("The data type " + value.GetDataType().ToString() + " is not supported. Only float or double is supported by CNTK.");
            }

            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);
            for (int seqIndex = 0, dataIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                var seqData = new List<T>();
                // Todo: consider mask
                // Todo: make it more efficient.
                for (int i = 0; i < numOfElementsInSample * numOfSamplesInSequence; i++)
                {
                    seqData.Add(outputData[dataIndex++]);
                }
                data.Add(seqData);
            }
        }

        // The value represents a n-dimensional tensor with 2 dynamic axes: sequence and batch
        public static void CopyTo<T>(this Value value, Variable variable, List<List<uint>> data)
        {
            throw new NotImplementedException("Not implemented");
        }

        // The value represents a n-dimensional tensor with 2 dynamic axes: sequence and batch
        public static void CopyTo<T>(this Value value, Variable variable, List<List<T>> data, List<List<uint>> indexes, List<List<uint>> nnzCounts)
        {
            throw new NotImplementedException("Not implemented");
        }
        public static void CopyTo<T>(this Value value, Variable variable, List<Sequence<T>> data)
        {
            // Todo: can a value only have the sequencee axis, but no batch axis??
            if (value.Shape.SubShape(0, value.Shape.Rank - 2) != variable.Shape)
            {
                throw new ArgumentException("The variable and value does not have same shape.");
            }

            var rawData = new List<List<T>>();
            CopyTo(value, variable, rawData);
            // Todo: optimize to avoid data copy
            foreach (var l in rawData)
            {
                var seq = new Sequence<T>(variable.Shape);
                seq.AddRange(l);
                data.Add(seq);
            }
        }

        public static void CopyTo<T>(this Value value, Variable variable, List<SequenceOneHotVector> data)
        {
            // Todo: can a value only have the sequencee axis, but no batch axis??
            if (value.Shape.SubShape(0, value.Shape.Rank - 2) != variable.Shape)
            {
                throw new ArgumentException("The variable and value does not have same shape.");
            }

            if (variable.Shape.Rank > 1)
            {
                throw new System.ArgumentException("The OneHot vector requires the variable has only 1 dimension.");
            }

            var vocabSize = variable.Shape[0];
            var rawData = new List<List<uint>>();
            CopyTo(value, variable, rawData);
            // Todo: optimize to avoid data copy
            foreach (var l in rawData)
            {
                var seq = new SequenceOneHotVector(vocabSize);
                seq.AddRange(l);
                data.Add(seq);
            }
        }

        public static void CopyTo<T>(this Value value, Variable variable, List<SequenceSparse<T>> data)
        {
            // Todo: can a value only have the sequencee axis, but no batch axis??
            if (value.Shape.SubShape(0, value.Shape.Rank - 2) != variable.Shape)
            {
                throw new ArgumentException("The variable and value does not have same shape.");
            }

            throw new NotImplementedException("Not implemented yet.");
         }
    }
}
