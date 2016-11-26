using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSEvalV2Example
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Load the model
            var myFunc = global::Function.LoadModel("01_OneHidden");

            // Ouput funciton info.
            var uid = myFunc.Uid();
            System.Console.WriteLine("Function id:" + (string.IsNullOrEmpty(uid) ? "(empty)" : uid));
            var name = myFunc.Name();
            System.Console.WriteLine("Function Name:" + (string.IsNullOrEmpty(name) ? "(empty)" : name));
            // Todo: directly return List() or use a wrapper?
            var argList = myFunc.Arguments().ToList();
            Console.WriteLine("Function arguments:");
            foreach (var arg in argList)
            {
                Console.WriteLine("    name=" + arg.Name() + ", kind=" + arg.Kind() + ", DataType=" + arg.GetDataType());
            }
            var outputList = myFunc.Outputs().ToList();
            Console.WriteLine("Function outputs:");
            foreach (var output in outputList)
            {
                Console.WriteLine("    name=" + output.Name() + ", kind=" + output.Kind() + ", DataType=" + output.GetDataType());
            }

            // prepare input for evaluation
            uint numOfSamples = 1;

            const string inputNodeName = "features";
            var inputVar = argList.Where(variable => string.Equals(variable.Name(),inputNodeName)).FirstOrDefault();          
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
            var outputVar = outputList.Where(variable => string.Equals(variable.Name(), outputNodeName)).FirstOrDefault();
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
    }
}
