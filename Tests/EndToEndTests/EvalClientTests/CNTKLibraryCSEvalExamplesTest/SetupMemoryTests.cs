//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// SetupMemoryTests.cs -- Setup Memory safety tests
//

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

        // Todo: move it to separate unit tests.
        public void NDArrayViewTest(DeviceDescriptor device)
        {
            Console.WriteLine("Test creating NDArrayView on devices.");

            var data = new float[10];
            var shape = new NDShape(1, 10);
            var n1 = new NDArrayView(shape, data, device);
            var n1Clone = n1.DeepClone(device);
            var n1CloneCPU = n1.DeepClone(DeviceDescriptor.CPUDevice);

            Console.WriteLine("n1: on " + n1.Device.AsString() + ", Storage:" + n1.StorageFormat + ", Shape:" + n1.Shape.AsString());
            Console.WriteLine("n1Clone: on " + n1Clone.Device.AsString() + ", Storage:" + n1Clone.StorageFormat + ", Shape:" + n1Clone.Shape.AsString());
            Console.WriteLine("n1CloneCPU: on " + n1CloneCPU.Device.AsString() + ", Storage:" + n1CloneCPU.StorageFormat + ", Shape:" + n1CloneCPU.Shape.AsString());

            int[] dimensions = { 4, 5 };
            var shape2 = NDShape.CreateNDShape(dimensions);
            float[] nonZeroValues = { 1, 5, 4, 2, 3, 9, 7, 8, 6 };
            int[] rowIndices = { 0, 2, 0, 1, 1, 3, 2, 2, 3 };
            int[] colStarts = { 0, 2, 4, 6, 7, 9};
            var s1 = new NDArrayView(shape2, colStarts, rowIndices, nonZeroValues, device, true);
            var s1Clone = s1.DeepClone(device);
            var s1DenseCPU = new NDArrayView(DataType.Float, StorageFormat.Dense, shape2, DeviceDescriptor.CPUDevice);
            s1DenseCPU.CopyFrom(s1);

            Console.WriteLine("s1: on " + s1.Device.AsString() + ", Storage:" + s1.StorageFormat + ", Shape:" + s1.Shape.AsString());
            Console.WriteLine("s1Clone: on " + s1Clone.Device.AsString() + ", Storage:" + s1Clone.StorageFormat + ", Shape:" + s1Clone.Shape.AsString());
            Console.WriteLine("s1DenseCPU: on " + s1DenseCPU.Device.AsString() + ", Storage:" + s1DenseCPU.StorageFormat + ", Shape:" + s1DenseCPU.Shape.AsString());
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
                Function modelFunc = Function.Load(modelFilePath, device);

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

                var imageList = new List<string>() { "00000.png", "00001.png", "00002.png" };
                foreach (var image in imageList)
                {
                    CNTKLibraryManagedExamples.ThrowIfFileNotExist(image, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", image));
                }
                Bitmap bmp, resized;
                List<float> resizedCHW;
                var seqData1 = new List<float>();
                var seqData2 = new List<float>();
                for (int sampleIndex = 0; sampleIndex < imageList.Count; sampleIndex++)
                {
                    bmp = new Bitmap(Bitmap.FromFile(imageList[sampleIndex]));
                    resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    resizedCHW = resized.ParallelExtractCHW();
                    if (sampleIndex < imageList.Count - 1)
                        seqData1.AddRange(resizedCHW);
                    seqData2.AddRange(resizedCHW);
                }

                var inputDataMap1 = new Dictionary<Variable, Value>();
                var outputDataMap1 = new Dictionary<Variable, Value>();
                var inputVal1 = Value.CreateBatch(inputVar.Shape, seqData1, device);
                inputDataMap1.Add(inputVar, inputVal1);
                outputDataMap1.Add(outputVar, null);

                // Using temprary Value object returned by Evaluate().
                modelFunc.Evaluate(inputDataMap1, outputDataMap1, device);
                var outputVal1 = outputDataMap1[outputVar];
                var outputData1 = outputVal1.GetDenseData<float>(outputVar);

                // Using cloned persistent Value object returned by Evaluate().
                var outputDataMap1WithClone = new Dictionary<Variable, Value>();
                outputDataMap1WithClone.Add(outputVar, null);
                modelFunc.Evaluate(inputDataMap1, outputDataMap1WithClone, true, device);

                // Using temprary Value object which overwrites the one returned by the previous Evaluate().
                var inputDataMap2 = new Dictionary<Variable, Value>();
                var outputDataMap2 = new Dictionary<Variable, Value>();
                var inputVal2 = Value.CreateBatch(inputVar.Shape, seqData2, device);
                inputDataMap2.Add(inputVar, inputVal2);
                outputDataMap2.Add(outputVar, null);
                modelFunc.Evaluate(inputDataMap2, outputDataMap2, device);

                // Test access to the persistent Value object, which should be still valid.
                var outputVal1WithClone = outputDataMap1WithClone[outputVar];
                var outputData1WithClone = outputVal1WithClone.GetDenseData<float>(outputVar);

                // Test access to the temprary Value object returned by the latest Evaluate().
                var outputVal2 = outputDataMap2[outputVar];
                var outputData2 = outputVal2.GetDenseData<float>(outputVar);

                // Test access to the temprary Value object returned by the previous Evaluate(), which is not valid any more.
                bool exceptionCaught = false;
                try 
                {
                    var data = outputVal1.GetDenseData<float>(outputVar);
                }
                catch (Exception ex)
                {
                    if (ex is ApplicationException && ex.Message.StartsWith("This Value object is invalid and can no longer be accessed."))
                        exceptionCaught = true;
                }
                if (exceptionCaught == false)
                {
                    throw new ApplicationException("The expected exception has not been caught.");
                }

                MemoryTests.OutputVal = outputVal1WithClone;

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
