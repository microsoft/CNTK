//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs : Tests of CNTK Library C# model training examples.
//
using CNTK;
using CNTK.CSTrainingExamples;
using System;
using System.Collections.Generic;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    class Program
    {
        static void Main(string[] args)
        {
            // Todo: move to a separate unit test.
            Console.WriteLine("Test CNTKLibraryCSTrainingExamples");
#if CPUONLY
            Console.WriteLine("======== Train model using CPUOnly build ========");
#else
            Console.WriteLine("======== Train model using GPU build ========");
#endif
            List<DeviceDescriptor> devices = new List<DeviceDescriptor>();
            if (ShouldRunOnCpu())
            {
                devices.Add(DeviceDescriptor.CPUDevice);
            }
            if (ShouldRunOnGpu())
            {
                devices.Add(DeviceDescriptor.GPUDevice(0));
            }

            string runTest = args.Length == 0 ? string.Empty : args[0];

            if (args.Length > 1)
            {
                Console.WriteLine($"-------- running with test data prefix : {args[1]} --------");
                TestCommon.TestDataDirPrefix = args[1];
            }
            else
            {
                Console.WriteLine("-------- No data folder path found in input, using default paths.");
                TestCommon.TestDataDirPrefix = "../../";
            }

            foreach (var device in devices)
            {
                // Data folders of example classes are set for non-CNTK test runs.
                // In case of CNTK test runs (runTest is set to a test name) data folders need to be set accordingly.
                switch (runTest)
                {
                    case "LogisticRegressionTest":
                        Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
                        LogisticRegression.TrainAndEvaluate(device);
                        break;
                    case "SimpleFeedForwardClassifierTest":
                        Console.WriteLine($"======== running SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier using {device.Type} ========");
                        SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
                        break;
                    case "CifarResNetClassifierTest":
                        Console.WriteLine($"======== running CifarResNet.TrainAndEvaluate using {device.Type} ========");
                        
                        if (args.Length > 1)
                        {
                            Console.WriteLine($"-------- running with test data in {args[1]} --------");
                            // this test uses data from external folder, we execute this test with full data dir.
                            CifarResNetClassifier.CifarDataFolder = TestCommon.TestDataDirPrefix;
                        }

                        CifarResNetClassifier.TrainAndEvaluate(device, true);
                        break;
                    case "LSTMSequenceClassifierTest":
                        Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
                        LSTMSequenceClassifier.Train(device);
                        break;
                    case "MNISTClassifierTest":
                        Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with Convnet using {device.Type} ========");
                        MNISTClassifier.TrainAndEvaluate(device, true, true);
                        break;
                    case "TransferLearningTest":
                        TransferLearning.BaseResnetModelFile = "ResNet_18.model";
                        Console.WriteLine($"======== running TransferLearning.TrainAndEvaluate with animal data using {device.Type} ========");
                        TransferLearning.TrainAndEvaluateWithAnimalData(device, true);
                        break;

                    case "":
                        RunAllExamples(device);
                        break;

                    default:
                        Console.WriteLine("'{0}' is not a valid test name.", runTest);
                        break;
                }
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

        static void RunAllExamples(DeviceDescriptor device)
        {
            Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
            LogisticRegression.TrainAndEvaluate(device);

            Console.WriteLine($"======== running SimpleFeedForwardClassifier.TrainSimpleFeedForwardClassifier using {device.Type} ========");
            SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate using {device.Type} with MLP classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate using {device.Type} with convolution neural network ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            if (device.Type == DeviceKind.GPU)
            {
                Console.WriteLine($"======== running CifarResNet.TrainAndEvaluate using {device.Type} ========");
                CifarResNetClassifier.TrainAndEvaluate(device, true);
            }

            if (device.Type == DeviceKind.GPU)
            {
                Console.WriteLine($"======== running TransferLearning.TrainAndEvaluateWithFlowerData using {device.Type} ========");
                TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

                Console.WriteLine($"======== running TransferLearning.TrainAndEvaluateWithAnimalData using {device.Type} ========");
                TransferLearning.TrainAndEvaluateWithAnimalData(device, true);
            }
            Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device);
        }
    }

    public static class TestCommon
    {
        public static string TestDataDirPrefix;
    }
}
