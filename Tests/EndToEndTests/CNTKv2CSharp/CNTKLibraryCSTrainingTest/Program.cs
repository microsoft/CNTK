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
using System.IO;
using System.Linq;
using System.Xml.Serialization;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    public class Program
    {
        static void PrintGraph(Function function, int spaces, bool useName = false)
        {
            string indent = new string('.', spaces);
            if (function.Inputs.Count() == 0)
            {
                Console.WriteLine(indent + "(" + (useName ? function.Name : function.Uid) + ")" +
                    "(" + function.OpName + ")" + function.AsString());
                return;
            }

            foreach (var input in function.Inputs)
            {
                Console.WriteLine(indent + "(" + (useName ? function.Name : function.Uid) + ")" +
                    "(" + function.OpName + ")" + "->" +
                    "(" + (useName ? input.Name : input.Uid) + ")" + input.AsString());
            }

            foreach (var input in function.Inputs)
            {
                if (input.Owner != null)
                {
                    Function f = input.Owner;
                    PrintGraph(f, spaces + 4, useName);
                }
            }
        }

        static private Function CreateModel(DeviceDescriptor device, int HiddenLayerCount, int HiddenLayerDimension, int OutputClassesCount, Variable Input)
        {

            Function[] HiddenLayers = new Function[HiddenLayerCount];
            for (int i = 0; i < HiddenLayerCount - 1; i++)
            {
                if (i == 0)
                    HiddenLayers[i] = TestHelper.Dense(Input, HiddenLayerDimension, device, Activation.Tanh, "");
                else
                    HiddenLayers[i] = TestHelper.Dense(HiddenLayers[i - 1], HiddenLayerDimension, device, Activation.Tanh, "");
            }

            return TestHelper.Dense(HiddenLayers[HiddenLayerCount - 2], OutputClassesCount, device, Activation.Sigmoid, "");
        }

        public class EvaResult
        {
            public EvaResult()
            { }

            public EvaResult(string line, float d0, float d1)
            {
                this.line = line;
                this.d0 = d0;
                this.d1 = d1;
            }
            public string line;
            public float d0;
            public float d1;
        }

        static void TestEval()
        {
            var inputDim = 9;
            var device = DeviceDescriptor.CPUDevice;

            //Function modelFunc = Function.Load(@"X:\_TLC\TLC_Misc\runs\CntkModel.0epochs.ag", device);
            // Function modelFunc = Function.Load(@"E:\LiqunWA\CNTKIssues\TLC\CNTKCSharp4Jignesh\CNTKCSharp4Jignesh\4.model\Predictor\CntkModel", device);
            Function modelFunc = Function.Load(@"..\TLC\CNTKCSharp4Jignesh\CNTKCSharp4Jignesh\4.model\Predictor\CntkModel", device);

            // The model has only one output.
            // You can also use the following way to get output variable by name:
            // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
            Variable inputVar = modelFunc.Arguments.Single();
            Variable outputVar = modelFunc.Output;

            string line;
            List<EvaResult> dictEvalResults = new List<EvaResult>();
            bool createRes = false;
            if (createRes)
            {
                System.IO.StreamReader file = new System.IO.StreamReader(@"E:\LiqunWA\CNTKIssues\TLC\CNTKCSharp4Jignesh\CNTKCSharp4Jignesh\breast-cancer-15k-noheader.txt");
                while ((line = file.ReadLine()) != null)
                {
                    //System.Console.WriteLine(line);
                    IList<IList<float>> outputData = EvalOneLine(inputDim, device, modelFunc, inputVar, outputVar, line);

                    //Console.WriteLine(outputData[0][1].ToString());
                    Console.WriteLine(string.Format("{0:G16}", outputData[0][1]));
                    dictEvalResults.Add(new EvaResult(line, outputData[0][0], outputData[0][1]));

                    //outputData[0][1]
                }
            }

            XmlSerializer serializer = new XmlSerializer(typeof(List<EvaResult>));
            // string resFile = @"E:\LiqunWA\CNTKIssues\TLC\CNTKCSharp4Jignesh\result.xml";
            string resFile = @"..\TLC\CNTKCSharp4Jignesh\result.xml";

            if (createRes)
            {
                using (TextWriter textWriter = new StreamWriter(resFile))
                {
                    serializer.Serialize(textWriter, dictEvalResults);

                    textWriter.Close();
                }
            }

            using (var myFileStream = new FileStream(resFile, FileMode.Open))
            {
                dictEvalResults = (List<EvaResult>)serializer.Deserialize(myFileStream);
            }

            for (int i = 0; i < 20; i++)
            {
                foreach (var l in dictEvalResults)
                {
                    IList<IList<float>> outputData = EvalOneLine(inputDim, device, modelFunc, inputVar, outputVar, l.line);
                    if (outputData[0][0] != l.d0|| outputData[0][1] != l.d1)
                    {
                        Console.WriteLine($"Eval mismatch: {String.Format("{0:F20}", outputData[0][0])} != {String.Format("{0:F20}", l.d0)} --- {String.Format("{0:F20}", outputData[0][1])} != {String.Format("{0:F20}", l.d1)}");
                    }
                }
                Console.WriteLine($"Pass {i}");
            }
        }

        private static IList<IList<float>> EvalOneLine(int inputDim, DeviceDescriptor device, Function modelFunc, Variable inputVar, Variable outputVar, string line)
        {
            var tokens = line.Split('\t');

            var tokensD = tokens.Select(x => float.Parse(x)).ToList();
            var features = tokensD.GetRange(1, inputDim);

            // Create input data map
            var inputDataMap = new Dictionary<Variable, Value>();
            var inputVal = Value.CreateBatch(new int[] { inputDim }, features, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create output data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            var outputVal = outputDataMap[outputVar];
            var outputData = outputVal.GetDenseData<float>(outputVar);
            return outputData;
        }

        static void Main(string[] args)
        {
            // TestEval();
            //int input_dim = 784, HiddenLayerCount = 2, HiddenLayerDimension = 400, OutputClassesCount = 10;
            //var Input = Variable.InputVariable(new int[] { input_dim }, DataType.Float);
            //Function d_model = CreateModel(DeviceDescriptor.CPUDevice, HiddenLayerCount, HiddenLayerDimension, OutputClassesCount, Input);

            //Function d_model = Function.Load("e:/dense.py.model", DeviceDescriptor.CPUDevice);
            // PrintGraph(d_model.RootFunction, 0);

            // CNTKModelEvaluator e = new CNTKModelEvaluator("E:/cntk/CNTK/x64/Debug/MNISTConvolution.model", 10);


            //var a = CNTKLib.InputVariable(NDShape.CreateNDShape(new[] { 1 }), CNTK.DataType.Double, name: "i");
            //var b = CNTKLib.InputVariable(NDShape.CreateNDShape(new[] { 1 }), CNTK.DataType.Double, name: "i");
            //var c = CNTKLib.Plus(a, b, name: "c");
            //var result = c.FindAllWithName("i");
            //var inputs = c.Inputs.Where(i => i.Name == "i");

            // Todo: move to a separate unit test.
            // Console.WriteLine("Test CNTKLibraryCSTrainingExamples");

            //var model = Function.Load(@"E:/cntk/CNTK/x64/Debug/MNISTConvolution.model", DeviceDescriptor.CPUDevice);
            //var m = model.Clone();
            //var conv1 = m.FindAllWithName("conv1").Last();
            //var dense1 = m.FindByName("dense1");
            //var dropout = m.FindByName("dropout");
            //m = Function.Combine(new[] { conv1.Output, dense1.Output, dropout.Output });

            //var inputLayer = m.Arguments.Single();
            //var outputLayers = m.Outputs;
            //var inputDataMap = new Dictionary<Variable, Value>();
            //var outputDataMap = new Dictionary<Variable, Value>();
            //foreach (var output in outputLayers)
            //{
            //    var arr = new float[output.Shape.TotalSize];
            //    var nd = new NDArrayView(output.Shape.Dimensions.Concat(new[] {1, 1 }).ToArray(), arr, DeviceDescriptor.CPUDevice);
            //    var v = new Value(nd);
            //    outputDataMap.Add(output, v);
            //}

            //var a = new float[784];
            //for (int j = 0; j < 10000; j++)
            //{
            //    inputDataMap.Clear();
            //    inputDataMap.Add(inputLayer, Value.CreateBatch<float>(inputLayer.Shape, a, DeviceDescriptor.CPUDevice));
            //    m.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
            //}

            //var x = CNTKLib.InputVariable(new int[]{ 10, 10, 2 }, DataType.Float, "input");
            //Function model = CNTKLib.Reshape(x, new int[] { NDShape.FreeDimension});
            //float[] data = new float[200];
            //Value v = Value.CreateBatch<float>(new int[] { 10, 10, 2 }, data, DeviceDescriptor.CPUDevice);
            //Dictionary<Variable, Value> o = new Dictionary<Variable, Value>() { { model.Output, null } };

            //model.Evaluate(
            //    new Dictionary<Variable, Value>() { { x, v} },
            //    o, 
            //    DeviceDescriptor.CPUDevice);

            //Parameter weights = new Parameter(
            //    new NDArrayView(new int[] { 2, 3 }, 
            //    new float[]{ 1, 2, 3, 4, 5, 6 }, 
            //    DeviceDescriptor.CPUDevice));
            var xVals = Value.CreateSequence<float>(new int[] { 2 }, new float[] { 1, 2, 3, 4, 5, 6 }, DeviceDescriptor.CPUDevice);
            NDArrayView v = xVals.Data;
            v.
            IList<IList<float>> data = xVals.GetDenseData<float>(Variable.InputVariable(new int[] { 2, 3 }, DataType.Float));

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

                
            foreach (var device in devices)
            {
                /// Data folders of example classes are set for non-CNTK test runs.
                /// In case of CNTK test runs (runTest is set to a test name) data folders need to be set accordingly.
                switch (runTest)
                {
                    case "LogisticRegressionTest":
                        Console.WriteLine($"======== running LogisticRegression.TrainAndEvaluate using {device.Type} ========");
                        LogisticRegression.TrainAndEvaluate(device);
                        break;
                    case "SimpleFeedForwardClassifierTest":
                        SimpleFeedForwardClassifierTest.DataFolder = ".";
                        Console.WriteLine($"======== running SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier using {device.Type} ========");
                        SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);
                        break;
                    case "CifarResNetClassifierTest":
                        CifarResNetClassifier.CifarDataFolder = "./cifar-10-batches-py";
                        Console.WriteLine($"======== running CifarResNet.TrainAndEvaluate using {device.Type} ========");
                        CifarResNetClassifier.TrainAndEvaluate(device, true);
                        break;
                    case "LSTMSequenceClassifierTest":
                        LSTMSequenceClassifier.DataFolder = "../../../Text/SequenceClassification/Data";
                        Console.WriteLine($"======== running LSTMSequenceClassifier.Train using {device.Type} ========");
                        LSTMSequenceClassifier.Train(device);
                        break;
                    case "MNISTClassifierTest":
                        MNISTClassifier.ImageDataFolder = "../../../Image/Data/";
                        Console.WriteLine($"======== running MNISTClassifier.TrainAndEvaluate with Convnet using {device.Type} ========");
                        MNISTClassifier.TrainAndEvaluate(device, true, true);
                        break;
                    case "TransferLearningTest":
                        TransferLearning.ExampleImageFoler = ".";
                        TransferLearning.BaseResnetModelFile = "ResNet_18.model";
                        Console.WriteLine($"======== running TransferLearning.TrainAndEvaluate with animal data using {device.Type} ========");
                        TransferLearning.TrainAndEvaluateWithAnimalData(device, true);
                        break;
                    default:
                        RunAllExamples(device);
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
}
