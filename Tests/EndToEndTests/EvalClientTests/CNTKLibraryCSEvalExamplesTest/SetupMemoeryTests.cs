using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    internal sealed class SetupMemoeryTests : IDisposable
    {
        public void Dispose()
        {
        }

        public void SetupUsingResetModel(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Setup memory tests using Resnet Model =====");

                var deviceList = DeviceDescriptor.AllDevices();
                MemoryTests.Device0 = deviceList[0];

                // Load the model.
                string modelFilePath = "resnet20.dnn";
                CNTKLibraryManagedExamples.ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                Function modelFunc = Function.LoadModel(modelFilePath, device);

                Variable inputVar = modelFunc.Arguments.Single();
                MemoryTests.ArgumentVar0 = inputVar;
                MemoryTests.InputVar0 = modelFunc.Inputs.First();

                MemoryTests.OutputVar0 = modelFunc.Outputs[0];
                MemoryTests.OutputVar = modelFunc.Output;
                Variable outputVar = MemoryTests.OutputVar;

                MemoryTests.Axis0 = outputVar.DynamicAxes.FirstOrDefault();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];
                int imageChannels = inputShape[2];
                int imageSize = inputShape.TotalSize;
                var inputDataMap = new Dictionary<Variable, Value>();
                var outputDataMap = new Dictionary<Variable, Value>();

                var imageList = new List<string>() { "00000.png", "00001.png", "00002.png" };
                foreach (var image in imageList)
                {
                    CNTKLibraryManagedExamples.ThrowIfFileNotExist(image, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", image));
                }
                Bitmap bmp, resized;
                List<float> resizedCHW;
                var seqData = new List<float>();
                for (int sampleIndex = 0; sampleIndex < imageList.Count; sampleIndex++)
                {
                    bmp = new Bitmap(Bitmap.FromFile(imageList[sampleIndex]));
                    resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    resizedCHW = resized.ParallelExtractCHW();
                    seqData.AddRange(resizedCHW);
                }

                var inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);
                inputDataMap.Add(inputVar, inputVal);
                outputDataMap.Add(outputVar, null);
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                MemoryTests.OutputVal = outputVal;

                Console.WriteLine("\nTest object reference inside SetupUsingResetModel.\n");
                MemoryTests.WriteOutputs();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }
    }
}
