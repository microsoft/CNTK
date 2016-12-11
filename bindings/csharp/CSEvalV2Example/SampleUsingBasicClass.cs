//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// SampleUsingBasicClass.cs -- Samples use low level classes for evaluation.
//
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CSEvalV2Example
{
    //
    // This example shows how to evaluate model using low-level APIs for evaluation.
    //
    public class SampleUsingBasicClass
    {
        public static void EvaluateV1ModelUsingNDView()
        {
            // Load the model
            var myFunc = Function.LoadModel("01_OneHidden");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            const string inputNodeName = "features";
            var inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputNodeName)).Single();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape.TotalSize * numOfSamples;
            float[] inputData = new float[numOfInputData];
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[i] = (float)(i % 255);
            }

            // Todo: create value directly from data.
            var dynamicAxisShape = new SizeTVector() { 1, numOfSamples };
            var inputShape = inputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));
            var inputNDArrayView = new NDArrayView(inputShape, inputData, numOfInputData, DeviceDescriptor.CPUDevice);
            var inputValue = new Value(inputNDArrayView);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            var outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();
            var outputShape = outputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));

            // Create output buffer
            // Todo: use the system created buffer?
            uint numOfOutputData = outputVar.Shape.TotalSize * numOfSamples;
            float[] outputData = new float[numOfOutputData];
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                outputData[i] = (float)0.0;
            }
            var outputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice);
            var outputValue = new Value(outputNDArrayView);

            // Create ouput map
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, outputValue);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        public static void EvaluateV2ModelUsingNDView()
        {
            // Load the model
            var myFunc = Function.LoadModel("z.model");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            // The z.model has only one input
            var inputVar = myFunc.Arguments.Single();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape.TotalSize * numOfSamples;
            float[] inputData = new float[numOfInputData];
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[i] = (float)(i % 255);
            }

            // Todo: create value directly from data.
            var dynamicAxisShape = new SizeTVector() { 1, numOfSamples };
            var inputShape = inputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));
            var inputNDArrayView = new NDArrayView(inputShape, inputData, numOfInputData, DeviceDescriptor.CPUDevice);
            var inputValue = new Value(inputNDArrayView);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // The z.model has only one output.
            var outputVar = myFunc.Output;
            var outputShape = outputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));

            // Create output buffer
            // Todo: use the system created buffer?
            uint numOfOutputData = outputVar.Shape.TotalSize * numOfSamples;
            float[] outputData = new float[numOfOutputData];
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                outputData[i] = (float)0.0;
            }
            var outputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice);
            var outputValue = new Value(outputNDArrayView);

            // Create ouput map
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, outputValue);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        public static void EvaluateUsingSystemAllocatedMemory()
        {
            // Load the model
            var myFunc = Function.LoadModel("z.model");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            // Only one input for the model.
            var inputVar = myFunc.Arguments.First();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape.TotalSize * numOfSamples;
            float[] inputData = new float[numOfInputData];
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData[i] = (float)(i % 255);
            }

            // Todo: create value directly from data.
            var dynamicAxisShape = new SizeTVector() { 1, numOfSamples };
            var inputShape = inputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));
            var inputNDArrayView = new NDArrayView(inputShape, inputData, numOfInputData, DeviceDescriptor.CPUDevice);
            var inputValue = new Value(inputNDArrayView);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // Prepare output. The model has only one output.
            var outputVar = myFunc.Output;

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // Get output value after evaluation
            var outputValue = outputMap[outputVar];
            var outputNDArrayView = outputValue.Data();
            var outputShape = outputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));

            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer if it is on CPU?
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize;
            float[] outputData = new float[numOfOutputData];
            var cpuOutputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice);
            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        public static void EvaluateUsingCreateValue()
        {
            // Load the model
            var myFunc = Function.LoadModel("01_OneHidden");

            // Ouput funciton info.
            OutputFunctionInfo(myFunc);

            // prepare input for evaluation
            uint numOfSamples = 1;

            const string inputNodeName = "features";
            var inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputNodeName)).Single();
            // Todo: get size directly from inputVar.
            uint numOfInputData = inputVar.Shape.TotalSize;
            var inputData = new List<float>();
            for (uint i = 0; i < numOfInputData; ++i)
            {
                inputData.Add(i % 255);
            }

            var inputVector = new FloatVector(inputData);
            var data = new FloatVectorVector() { inputVector };
            // Create value directly from data.
            var inputValue = Value.CreateDenseFloat(inputVar.Shape, data, DeviceDescriptor.CPUDevice);

            // Create input map
            // Todo: create a Dictionary wrapper?
            var inputMap = new UnorderedMapVariableValuePtr();
            inputMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            var outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new UnorderedMapVariableValuePtr();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // Get output value after evaluation
            var outputValue = outputMap[outputVar];
            var outputNDArrayView = outputValue.Data();

            var dynamicAxisShape = new SizeTVector() { 1, numOfSamples };
            var outputShape = outputVar.Shape.AppendShape(new NDShape(dynamicAxisShape));

            // Copy the data from the output buffer.
            // Todo: directly access the data in output buffer if it is on CPU?
            uint numOfOutputData = outputNDArrayView.Shape().TotalSize;
            float[] outputData = new float[numOfOutputData];
            var cpuOutputNDArrayView = new NDArrayView(outputShape, outputData, numOfOutputData, DeviceDescriptor.CPUDevice);
            cpuOutputNDArrayView.CopyFrom(outputNDArrayView);

            // Output results
            Console.WriteLine("Evaluation results:");
            for (uint i = 0; i < numOfOutputData; ++i)
            {
                Console.WriteLine(outputData[i]);
            }
        }

        private static void OutputFunctionInfo(Function func)
        {
            var uid = func.Uid;
            System.Console.WriteLine("Function id:" + (string.IsNullOrEmpty(uid) ? "(empty)" : uid));
            var name = func.Name;
            System.Console.WriteLine("Function Name:" + (string.IsNullOrEmpty(name) ? "(empty)" : name));

            // Todo: directly return List() or use a wrapper?
            var argList = func.Arguments.ToList();
            Console.WriteLine("Function arguments:");
            foreach (var arg in argList)
            {
                Console.WriteLine(string.Format("    name={0}, kind={1}, DataType={2}, TotalSize={3}", arg.Name, arg.Kind(), arg.GetDataType(), arg.Shape.TotalSize));
            }

            var outputList = func.Outputs.ToList();
            Console.WriteLine("Function outputs:");
            foreach (var output in outputList)
            {
                Console.WriteLine(string.Format("    name={0}, kind={1}, DataType={2}, TotalSize={3}", output.Name, output.Kind(), output.GetDataType(), output.Shape.TotalSize));
            }
        }
    }
}
