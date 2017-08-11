//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs : Tests of CNTK Library C# model training examples.
//
using CNTK;
using System;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    class Program
    {
        static void Main(string[] args)
        {
            // Todo: move to a separate unit test.
            Console.WriteLine("Test CNTKLibraryCSTrainingExamples");
#if CPUONLY
            Console.WriteLine("======== Train model on CPU using CPUOnly build ========");
#else
            Console.WriteLine("======== Train model on CPU using GPU build ========");
#endif

            if (ShouldRunOnCpu())
            {
                var device = DeviceDescriptor.CPUDevice;

                SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
            }

            if (ShouldRunOnGpu())
            {
                Console.WriteLine(" ====== Train model on GPU =====");
                var device = DeviceDescriptor.GPUDevice(0);

                SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
            }

            Console.WriteLine("======== Train completes. ========");
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
