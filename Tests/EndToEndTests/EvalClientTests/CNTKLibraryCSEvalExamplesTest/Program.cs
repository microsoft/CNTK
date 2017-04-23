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
                var device = DeviceDescriptor.CPUDevice;

                CNTKLibraryManagedExamples.EvaluationSingleImage(device);
                // Run memory tests.
                MemoryTests.ValidateObjectReferences(device);
                CNTKLibraryManagedExamples.EvaluationBatchOfImages(device);
                CNTKLibraryManagedExamples.EvaluateMultipleImagesInParallel(device);
                // Run memory tests again.
                MemoryTests.ValidateObjectReferences(device);

                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(device);
                CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(device);
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingSparse(device);
                // It is sufficient to test loading model from memory buffer only on CPU.
                CNTKLibraryManagedExamples.LoadModelFromMemory(device);

                MemoryTests.WriteOutputs();
            }

            if (ShouldRunOnGpu())
            {
                Console.WriteLine(" ====== Evaluate model on GPU =====");
                var device = DeviceDescriptor.GPUDevice(0);
                // Run memory tests.
                MemoryTests.ValidateObjectReferences(device);
                CNTKLibraryManagedExamples.EvaluationSingleImage(device);
                CNTKLibraryManagedExamples.EvaluationBatchOfImages(device);
                CNTKLibraryManagedExamples.EvaluateMultipleImagesInParallel(device);
                // Run memory tests.
                MemoryTests.ValidateObjectReferences(device);

                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(device);
                CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(device);
                CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingSparse(device);

                // Run memory tests again.
                MemoryTests.WriteOutputs();
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
