//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs -- Tests of CNTK Library C# Eval examples.
//

using System;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    class Program
    {
        private static bool isGPUDeviceAvailable;
        private static bool isInitialized = false;

        static void Main(string[] args)
        {
#if CPUONLY
            Console.WriteLine("======== Evaluate model on CPU using CPUOnly build ========");
#else
            Console.WriteLine("======== Evaluate model on CPU using GPU build ========");
#endif

            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.CPUDevice);
            CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.CPUDevice);

            if (IsGPUAvailable())
            {
                Console.WriteLine(" ====== Evaluate model on GPU =====");
                CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.GPUDevice(0));
            }

            Console.WriteLine("======== Evaluation completes. ========");
        }

        static bool IsGPUAvailable()
        {
            if (!isInitialized)
            {
#if CPUONLY
                isGPUDeviceAvailable = false;
#else
                string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

                // Check the environment variable TEST_DEVICE to decide whether to run on a CPU-only device.
                if (!string.IsNullOrEmpty(testDeviceSetting) && string.Equals(testDeviceSetting.ToLower(), "cpu"))
                {
                    isGPUDeviceAvailable = false;
                }
                else
                {
                    isGPUDeviceAvailable = true;
                }
#endif
                isInitialized = true;
            }

            return isGPUDeviceAvailable;
        }
    }
}
