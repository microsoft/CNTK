//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CSEvalV2Library -- C# Eval V2 Library
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace CNTK
{
    public sealed class Evaluation
    {
        public Evaluation()
        {
            rootFunction = null;
        }

        // Todo: do we need DeviceDecriptor or a simplified version?
        public void LoadModel(string rootFunctionFile, DeviceDescriptor computeDevice)
        {
            if (!File.Exists(rootFunctionFile))
            {
                throw new FileNotFoundException(string.Format("File '{0}' not found.", rootFunctionFile));
            }
            Function.LoadModel(rootFunctionFile, computeDevice);
        }

        // Todo: use ulong, uint or int?
        public IDictionary<string, List<uint>> InputsDimensions
        {
            get
            {
                return GetNodeShapes(VariableKind.Input);
            }
        }

        public IDictionary<string, List<uint>> OutputsDimensions
        {
            get
            {
                return GetNodeShapes(VariableKind.Output);
            }
        }
        
        // Todo: Size_t to ulong/uint/init?? List.Count is int
        public IDictionary<string, uint> InputsSize
        {
            get
            {
                return GetNodeSizes(VariableKind.Input);
            }
        }

        public IDictionary<string, uint> OutputsSize
        {
            get
            {
                return GetNodeSizes(VariableKind.Output);
            }
        }

        //
        // Set output Variable
        //
        // Todo: inputVariable as ref/out?
        public void SetEvaluationOutput(List<string> outputVariables, List<string> inputVariables)
        {

        }

        // Todo: set default parameters  = DeviceDescriptor.UseDefaultDevice()
        // Todo: how to define DeviceDescriptor?? Better just enum or something easier
        // Todo: use Variable as method, string as extension method.
        public void Evaluate(Dictionary<string, Value> arguments, Dictionary<string, Value> outputs, DeviceDescriptor computeDevice)
        {
            if (rootFunction == null)
            {
                throw new NullReferenceException("No rootFunction is loaded. Please load the rootFunction first before evaluation.");
            }

            // Evaluate the rootFunction.
            var argMap = new UnorderedMapVariableValuePtr();
            foreach (var p in arguments)
            {
                var variable = rootFunction.Arguments.Where(v => string.Equals(v.Name, p.Key)).FirstOrDefault();
                if (variable == null)
                {
                    throw new KeyNotFoundException("No input variable '" + p.Key + "' found.");
                }
                argMap.Add(variable, p.Value);
            }

            var outMap = new UnorderedMapVariableValuePtr();
            foreach (var p in outputs)
            {
                var variable = rootFunction.Outputs.Where(v => string.Equals(v.Name, p.Key)).FirstOrDefault();
                if (variable == null)
                {
                    throw new KeyNotFoundException("No output variable '" + p.Key + "' found.");
                }
                outMap.Add(variable, p.Value);
            }

            rootFunction.Evaluate(argMap, outMap, computeDevice);

            foreach (var p in outMap)
            {
                outputs[p.Key.Name] = p.Value;
            }
        }

        // Create Value based on dense input
        // Todo: could this be a extension to Value class??
        // Todo: use Variable instead of varName. VarName as extension method
        // Todo: List can have maximal 2^31-1, enough? Otherwise need to go to array which supports 64bit size
        public Value CreateValue<T>(string varName, List<List<T>> sequences, DeviceDescriptor computeDevice)
        {
            var variable = getVariableByName(varName);
            var dim = variable.Shape.TotalSize;

            if (typeof(T).Equals(typeof(float)))
            {
                var inputSeqVector = new FloatVectorVector();
                foreach (var seq in sequences)
                {
                    if (seq.Count() % dim != 0)
                    {
                        throw new InvalidDataException("the number of data in sequences does not match the input dimension");
                    }
                    var samples = new FloatVector(seq);
                    inputSeqVector.Add(samples);
                }
                var inputValue = Value.CreateDenseFloat(variable.Shape, inputSeqVector, computeDevice);
                return inputValue;
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                var inputSeqVector = new DoubleVectorVector();
                foreach (var seq in sequences)
                {
                    if (seq.Count() % dim != 0)
                    {
                        throw new InvalidDataException("the number of data in sequences does not match the input dimension");
                    }
                    var samples = new DoubleVector(seq);
                    inputSeqVector.Add(samples);
                }
                var inputValue = Value.CreateDenseDouble(variable.Shape, inputSeqVector, computeDevice);
                return inputValue;
            }
            else
            {
                throw new InvalidDataException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        // Create Value based on sparse input
        // Todo: could this be a extension to Value class??
        // Todo: use Variable instead of varName. VarName as extension method
        public Value CreateValue<T>(string varName, List<List<T>> data, List<List<long>> indexes, List<List<long>> nnzCounts, DeviceDescriptor computeDevice)
        {
            throw new NotImplementedException("Not implemented");
        }

        // Create Value based on onehot input
        // Todo: could this be a extension to Value class??
        // Todo: use Variable instead of varName. VarName as extension method
        public Value CreateValue<T>(string varName, List<List<long>> oneHotIndex, DeviceDescriptor computeDevice)
        {
            throw new NotImplementedException("Not implemented");
        }


        // Copy Value to List<List<T>> for dense input
        // Todo: could this be a extension to Value class??
        public void CopyValueTo<T>(string varName, Value value, List<List<T>> sequences)
        {
            // Todo: deal with GPUDevice.
            if (value.Device != DeviceDescriptor.CPUDevice)
            {
                throw new InvalidOperationException("Currently only CPU device is supported.");
            }

            if ((value.GetDataType() == DataType.Float) && (!typeof(T).Equals(typeof(float))) || 
                (value.GetDataType() == DataType.Double) && (!typeof(T).Equals(typeof(double))))
            {
                throw new InvalidDataException("The value type does not match the list type.");
            }

            // Todo: transform sparse to dense
            // Currently only for dense
            if ((value.GetStorageFormat() != StorageFormat.Dense))
            {
                throw new InvalidDataException("The value is not in denst format.");
            }

            var variable = getVariableByName(varName);
            var outputNDArrayView = value.Data();
            var outputShape = outputNDArrayView.Shape();

            var varRank = variable.Shape.Rank;
            var valueRank = outputNDArrayView.Shape().Rank;

            Debug.Assert(varRank + 2 == valueRank);
            var numOfElementsInSample = variable.Shape.TotalSize;
            var numOfSamplesInSequence = outputShape.GetDimensionSize(varRank);
            var numOfSequences = outputShape.GetDimensionSize(varRank+1);

            //var outputShape = outputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));

            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer?
            // Todo: need to map DataBuffer() to C#
            NDArrayView cpuOutputNDArrayView;
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize;
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
                throw new InvalidDataException("The data type " + value.GetDataType().ToString() + " is not supported. Only float or double is supported by CNTK.");
            }

            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);
            for (int seqIndex = 0, dataIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                var seqData = new List<T>();
                // Todo: make it more efficient.
                for (int i = 0; i < numOfElementsInSample * numOfSamplesInSequence; i++)
                {
                    seqData.Add(outputData[dataIndex++]);
                }
                sequences.Add(seqData);
            }
        }

        // Copy Value to List<List<T>> for sparse input
        public void CopyValueTo<T>(string varName, Value value, List<List<T>> data, List<List<uint>> indexes, List<List<uint>> nnzCounts)
        {
            throw new NotImplementedException("Not implemented");
        }

        public void CopyValueTo<T>(string varName, Value value, List<List<uint>> oneHotIndex)
        {
            throw new NotImplementedException("Not implemented");
        }

        public Evaluation Clone()
        {
            throw new NotImplementedException("Not implemented");
        }

        public Variable getVariableByName(string name)
        {
            var v = rootFunction.Arguments.Where(variable => string.Equals(variable.Name, name)).FirstOrDefault();
            if (v == null)
            {
                v = rootFunction.Outputs.Where(variable => string.Equals(variable.Name, name)).FirstOrDefault();
            }

            return v;
        }

        private Function rootFunction;

        private IDictionary<string, List<uint>> GetNodeShapes(VariableKind nodeKind)
        {
            var retVal = new Dictionary<string, List<uint>>();

            IEnumerable<Variable> varList;
            if (nodeKind == VariableKind.Input)
            {
                varList = rootFunction.Arguments;
            }
            else if (nodeKind == VariableKind.Output)
            {
                varList = rootFunction.Outputs;
            }
            else 
            {
                // Todo: Use nameof after VS2015.
                throw new ArgumentException("Node kind must be '" + "VariableKind.Input" + "' or '" + "VariableKind.Output" + "'.");
            }

            foreach (var arg in varList)
            {
                if (retVal.ContainsKey(arg.Name))
                {
                    throw new Exception("duplicated name '" + arg.Name + "'.");
                }
                var dim = new List<uint>();
                // The Dimensions is IEnumerable<uint>
                // Todo: fix the swig to output IEnumberable<ulong>
                foreach (var d in arg.Shape.Dimensions)
                {
                    dim.Add(d);
                }
                retVal.Add(arg.Name, dim);
            }

            return retVal;
        }

        private IDictionary<string, uint> GetNodeSizes(VariableKind nodeKind)
        {
            var retVal = new Dictionary<string, uint>();

            IEnumerable<Variable> varList;
            if (nodeKind == VariableKind.Input)
            {
                varList = rootFunction.Arguments;
            }
            else if (nodeKind == VariableKind.Output)
            {
                varList = rootFunction.Outputs;
            }
            else
            {
                // Todo: Use nameof after VS2015.
                throw new ArgumentException("Node kind must be '" + "VariableKind.Input" + "' or '" + "VariableKind.Output" + "'.");
            }

            foreach (var arg in varList)
            {
                if (retVal.ContainsKey(arg.Name))
                {
                    throw new Exception("duplicated name '" + arg.Name + "'.");
                }

                retVal.Add(arg.Name, arg.Shape.TotalSize);
            }

            return retVal;
        }

    }
}
