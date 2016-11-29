//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// program.cs -- Example for using C# Eval V2 API.
//

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSEvalV2Example
{
    public class Program
    {
        static void EvaluateV1ModelUsingNDView()
        {
            // Load the model
            var myFunc = global::Function.LoadModel("01_OneHidden");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            const string inputNodeName = "features";
            var inputVar = myFunc.Arguments().Where(variable => string.Equals(variable.Name(), inputNodeName)).FirstOrDefault();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape().TotalSize() * numOfSamples;
            float[] inputData = new float[numOfInputData];
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[i] = (float)(i % 255);
            }

            // Todo: create value directly from data.
            var dynamicAxisShape = new global::SizeTVector() { 1, numOfSamples };
            var inputShape = inputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));
            var inputNDArrayView = new NDArrayView(inputShape, inputData, numOfInputData, DeviceDescriptor.CPUDevice());
            var inputValue = new Value(inputNDArrayView);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            var outputVar = myFunc.Outputs().Where(variable => string.Equals(variable.Name(), outputNodeName)).FirstOrDefault();
            var outputShape = outputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));

            // Create output buffer
            // Todo: use the system created buffer?
            uint numOfOutputData = outputVar.Shape().TotalSize() * numOfSamples;
            float[] outputData = new float[numOfOutputData];
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                outputData[i] = (float)0.0;
            }
            var outputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice());
            var outputValue = new Value(outputNDArrayView);

            // Create ouput map
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, outputValue);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice());

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        static void EvaluateV2ModelUsingNDView()
        {
            // Load the model
            var myFunc = global::Function.LoadModel("z.model");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            // The z.model has only one input
            var inputVar = myFunc.Arguments().FirstOrDefault();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape().TotalSize() * numOfSamples;
            float[] inputData = new float[numOfInputData];
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[i] = (float)(i % 255);
            }

            // Todo: create value directly from data.
            var dynamicAxisShape = new global::SizeTVector() { 1, numOfSamples };
            var inputShape = inputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));
            var inputNDArrayView = new NDArrayView(inputShape, inputData, numOfInputData, DeviceDescriptor.CPUDevice());
            var inputValue = new Value(inputNDArrayView);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // The z.model has only one output.
            var outputVar = myFunc.Output();
            var outputShape = outputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));

            // Create output buffer
            // Todo: use the system created buffer?
            uint numOfOutputData = outputVar.Shape().TotalSize() * numOfSamples;
            float[] outputData = new float[numOfOutputData];
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                outputData[i] = (float)0.0;
            }
            var outputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice());
            var outputValue = new Value(outputNDArrayView);

            // Create ouput map
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, outputValue);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice());

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        static void EvaluateUsingSystemAllocatedMemory()
        {
            // Load the model
            var myFunc = global::Function.LoadModel("z.model");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            // Only one input for the model.
            var inputVar = myFunc.Arguments().First(); 
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape().TotalSize() * numOfSamples;
            float[] inputData = new float[numOfInputData];
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[i] = (float)(i % 255);
            }

            // Todo: create value directly from data.
            var dynamicAxisShape = new global::SizeTVector() { 1, numOfSamples };
            var inputShape = inputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));
            var inputNDArrayView = new NDArrayView(inputShape, inputData, numOfInputData, DeviceDescriptor.CPUDevice());
            var inputValue = new Value(inputNDArrayView);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // Prepare output. The model has only one output.
            var outputVar = myFunc.Output();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice());

            // Get output value after evaluation
            var outputValue = outputMap[outputVar];
            var outputNDArrayView = outputValue.Data();
            var outputShape = outputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));
            
            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer if it is on CPU?
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize();
            float[] outputData = new float[numOfOutputData];
            var cpuOutputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice());
            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        static void EvaluateUsingCreateValue()
        {
            // Load the model
            var myFunc = global::Function.LoadModel("01_OneHidden");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            const string inputNodeName = "features";
            var inputVar = myFunc.Arguments().Where(variable => string.Equals(variable.Name(), inputNodeName)).FirstOrDefault();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape().TotalSize();
            var inputData = new List<float>();
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData.Add(i % 255);
            }

            var inputVector = new FloatVector(inputData); 
            var data = new FloatVectorVector() {inputVector};
            // Create value directly from data.
            var inputValue = Value.CreateDenseFloat(inputVar.Shape(), data, DeviceDescriptor.CPUDevice());

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            var outputVar = myFunc.Outputs().Where(variable => string.Equals(variable.Name(), outputNodeName)).FirstOrDefault();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice());

            // Get output value after evaluation
            var outputValue = outputMap[outputVar];
            var outputNDArrayView = outputValue.Data();

            var dynamicAxisShape = new global::SizeTVector() { 1, numOfSamples };
            var outputShape = outputVar.Shape().AppendShape(new NDShape(dynamicAxisShape));

            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer if it is on CPU?
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize();
            float[] outputData = new float[numOfOutputData];
            var cpuOutputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice());
            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }


        static void EvaluateUsingCSEvalLib()
        {
            // Load the model
            var model = new CNTK.CSharp.Evaluation();

            model.LoadModel("z.model", DeviceDescriptor.CPUDevice());

            // prepare input for evaluation
            int numOfSamples = 1;

            var inputDims = model.GetInputSizes();
            const string inputNodeName = "features";
            
            ulong numOfInputData = inputDims[inputNodeName];
            var inputData = new List<List<float>>(numOfSamples);
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[0].Add(i % 255);
            }

            var inputValue = model.CreateValue<float>(inputNodeName, inputData, DeviceDescriptor.CPUDevice());

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new Dictionary<string, Value>() { { inputNodeName, inputValue } }; 

            // Prepare output
            const string outputNodeName = "out.z_output";

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<string, Value>() { { outputNodeName, null } };

            // Evalaute
            // Todo: test on GPUDevice()?
            model.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice());

            var output = new List<List<float>>();
            var outputDims = model.GetOutputSizes();
            ulong numOfElementsInSample = outputDims[outputNodeName];

            model.CopyValueTo<float>(outputNodeName, outputMap[outputNodeName], output);

            ulong seqNo = 0;
            foreach (var seq in output)
            { 
                var numOfSamplesInSequence = (ulong)seq.Count/numOfElementsInSample;
                ulong elementIndex = 0;
                ulong sampleIndex = 0;
                Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                foreach (var data in seq)
                {
                    if (elementIndex++ == 0) 
                    {
                        Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    }
                    Console.Write(" " + data);
                    if (elementIndex == numOfElementsInSample)
                    {
                        Console.WriteLine(".");
                        elementIndex = 0;
                        sampleIndex++;
                    }
                }
                seqNo++;
            }
        }

        static void Main(string[] args)
        {
            
            //Console.WriteLine("======== Evaluate V1 Model ========");
            // EvaluateV1ModelUsingNDView();
            //Console.WriteLine("======== Evaluate V2 Model ========");
            //EvaluateV2ModelUsingNDView();
            //Console.WriteLine("======== Evaluate Model Using System Allocated Memory for Output Value ========");
            //EvaluateUsingSystemAllocatedMemory();
            //Console.WriteLine("======== Evaluate Using Value::Create ========");
            //EvaluateUsingCreateValue();
            Console.WriteLine("======== Evaluate Using EvalV2Library ========");
            EvaluateUsingCSEvalLib();
        }

        private static void OutputFunctionInfo(global::Function func)
        {
            var uid = func.Uid();
            System.Console.WriteLine("Function id:" + (string.IsNullOrEmpty(uid) ? "(empty)" : uid));
            var name = func.Name();
            System.Console.WriteLine("Function Name:" + (string.IsNullOrEmpty(name) ? "(empty)" : name));

            // Todo: directly return List() or use a wrapper?
            var argList = func.Arguments().ToList();
            Console.WriteLine("Function arguments:");
            foreach (var arg in argList)
            {
                Console.WriteLine("    name=" + arg.Name() + ", kind=" + arg.Kind() + ", DataType=" + arg.GetDataType() + ", TotalSize=" + arg.Shape().TotalSize());
            }

            var outputList = func.Outputs().ToList();
            Console.WriteLine("Function outputs:");
            foreach (var output in outputList)
            {
                Console.WriteLine("    name=" + output.Name() + ", kind=" + output.Kind() + ", DataType=" + output.GetDataType() + ", TotalSize=" + output.Shape().TotalSize());
            }
        }
    }
}
