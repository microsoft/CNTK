//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs : Tests of CNTK Library C# model training examples.
//
using CNTK;
using CNTK.CSTrainingExamples;
using CNTK.HighLevelAPI;
using System;
using System.Collections.Generic;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    class Program
    {
        static void CompareWithBuiltInTimes(Function times, DeviceDescriptor device, int outDim, int inDim)
        {
            int batchSize = 3;
            float[] inputData = new float[inDim* batchSize];
            for (int i = 0; i< inputData.Length; ++i)
                inputData[i] = 2.3F;

            var input = times.Arguments[0];
            var inputDataValue = Value.CreateBatch(input.Shape, inputData, device);

            float[] rootGradientData = new float[outDim * batchSize];
            for (int i = 0; i < rootGradientData.Length; ++i)
                rootGradientData[i] = 1.0F;
            
            var rootGradientValue = Value.CreateBatch(times.Output.Shape, rootGradientData, device);

            UnorderedMapVariableValuePtr arguments = new UnorderedMapVariableValuePtr();
            arguments.Add(new KeyValuePair<Variable, Value>(input, inputDataValue));
            UnorderedMapVariableValuePtr outputValues = new UnorderedMapVariableValuePtr();
            outputValues.Add(new KeyValuePair<Variable, Value>(times.Output, null));

            // UnorderedSet outputsToRetainBackwardStateFor = new UnorderedSet();
            // var backPropState = times.Forward(arguments, outputValues, device, outputsToRetainBackwardStateFor);
            var backPropState = times.Forward(arguments, outputValues, device);

            var parameter = times.Parameters()[0];

            UnorderedMapVariableValuePtr rootGradientValues = new UnorderedMapVariableValuePtr();
            rootGradientValues.Add(new KeyValuePair<Variable, Value>(times.Output, rootGradientValue));
            UnorderedMapVariableValuePtr backPropagatedGradientValuesForInputs = new UnorderedMapVariableValuePtr();
            backPropagatedGradientValuesForInputs.Add(new KeyValuePair<Variable, Value>(parameter, null));
            times.Backward(backPropState, rootGradientValues, backPropagatedGradientValuesForInputs);
            var userDefinedTimesOutputValue = outputValues[times.Output];
            var userDefinedTimesInputGradientValue = backPropagatedGradientValuesForInputs[parameter];

            Constant plusParam = new Constant(times.Output.Shape, DataType.Float, 3, device);
            var mixTimesPlus = CNTKLib.Plus(times, plusParam);
            mixTimesPlus.Evaluate(new Dictionary<Variable, Value>{ { input, inputDataValue } }, 
                new Dictionary<Variable, Value> { { mixTimesPlus.Output, null} }, device);
        }

        static void UserTimesFunctionExample()
        {
            DeviceDescriptor device = DeviceDescriptor.CPUDevice;
            int outDim = 15;
            int inDim = 10;
            Parameter W = new Parameter(new int[]{ outDim, inDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
            Variable x = Variable.InputVariable(new int[]{ inDim }, DataType.Float, "", new List<Axis>{ Axis.DefaultBatchAxis() }, false);
            VariableVector inputs = new VariableVector();
            inputs.Add(x);
            Function userDefinedTimes = UserTimesFunction.Create(W, x, "userDefinedTimes");

            CompareWithBuiltInTimes(userDefinedTimes, device, outDim, inDim);
        }

        static void Main(string[] args)
        {
            // Todo: move to a separate unit test.
            Console.WriteLine("Test CNTKLibraryCSTrainingExamples");
#if CPUONLY
            Console.WriteLine("======== Train model using CPUOnly build ========");
#else
            Console.WriteLine("======== Train model using GPU build ========");
#endif
            UserTimesFunctionExample();
            CifarResNetClassifier.TrainAndEvaluate(DeviceDescriptor.GPUDevice(0), true);
            return;

            //List<DeviceDescriptor> devices = new List<DeviceDescriptor>();
            //if (ShouldRunOnCpu())
            //{
            //    devices.Add(DeviceDescriptor.CPUDevice);
            //}
            //if (ShouldRunOnGpu())
            //{
            //    devices.Add(DeviceDescriptor.GPUDevice(0));
            //}

            //string runTest = args.Length == 0 ? string.Empty : args[0];

                
            //foreach (var device in devices)
            //{
            //    /// Data folders of example classes are set for non-CNTK test runs.
            //    /// In case of CNTK test runs (runTest is set to a test name) data folders need to be set accordingly.
            //    switch (runTest)
            //    {
            //        case "LogisticRegressionTest":
            //            Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
            //            LogisticRegression.TrainAndEvaluate(device);
            //            break;
            //        case "SimpleFeedForwardClassifierTest":
            //            SimpleFeedForwardClassifierTest.DataFolder = ".";
            //            Console.WriteLine($"======== running SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier using {device.Type} ========");
            //            SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
            //            break;
            //        case "CifarResNetClassifierTest":
            //            CifarResNetClassifier.CifarDataFolder = "./cifar-10-batches-py";
            //            Console.WriteLine($"======== running CifarResNet.TrainAndEvaluate using {device.Type} ========");
            //            CifarResNetClassifier.TrainAndEvaluate(device, true);
            //            break;
            //        case "LSTMSequenceClassifierTest":
            //            LSTMSequenceClassifier.DataFolder = "../../../Text/SequenceClassification/Data";
            //            Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
            //            LSTMSequenceClassifier.Train(device);
            //            break;
            //        case "MNISTClassifierTest":
            //            MNISTClassifier.ImageDataFolder = "../../../Image/Data/";
            //            Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with Convnet using {device.Type} ========");
            //            MNISTClassifier.TrainAndEvaluate(device, true, true);
            //            break;
            //        case "TransferLearningTest":
            //            TransferLearning.ExampleImageFoler = ".";
            //            TransferLearning.BaseResnetModelFile = "ResNet_18.model";
            //            Console.WriteLine($"======== running TransferLearning.TrainAndEvaluate with animal data using {device.Type} ========");
            //            TransferLearning.TrainAndEvaluateWithAnimalData(device, true);
            //            break;
            //        default:
            //            RunAllExamples(device);
            //            break;
            //    }
            //}

            //Console.WriteLine("======== Train completes. ========");
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
}
