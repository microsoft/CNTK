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
        static void Main(string[] args)
        {
#if CPUONLY
            Console.WriteLine("======== Evaluate model on CPU using CPUOnly build ========");
#else
            Console.WriteLine("======== Evaluate model on CPU using GPU build ========");
#endif

            if (ShouldRunOnCpu())
            {
                CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);
                CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);
                CNTKLibraryManagedExamples.EvaluateMultipleImagesInParallel(DeviceDescriptor.CPUDevice);
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.CPUDevice);
                CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.CPUDevice);
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingSparse(DeviceDescriptor.CPUDevice);
                // It is sufficient to test loading model from memory buffer only on CPU.
                CNTKLibraryManagedExamples.LoadModelFromMemory(DeviceDescriptor.CPUDevice);
            }

            if (ShouldRunOnGpu())
            {
                Console.WriteLine(" ====== Evaluate model on GPU =====");
                CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluateMultipleImagesInParallel(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.GPUDevice(0));
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingSparse(DeviceDescriptor.GPUDevice(0));
            }

            Console.WriteLine("======== Evaluation completes. ========");
        }

        static bool ShouldRunOnGpu()
        {
#if CPUONLY
            return false;
#else
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "gpu"));
#endif
        }

        static bool ShouldRunOnCpu()
        {
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "cpu"));
        }
    }
}
