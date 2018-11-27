using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CNTK.CSTrainingExamples
{
    /// <summary>
    /// This class shows how to do image classification using convolution network.
    /// It follows closely this CNTK Python tutorial:
    /// https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_201B_CIFAR-10_ImageHandsOn.ipynb
    /// </summary>
    public class CifarResNetClassifier
    {
        /// <summary>
        /// execution folder is: CNTK/x64/BuildFolder
        /// data folder is: CNTK/Examples/Image/DataSets
        /// </summary>
        public static string CifarDataFolder;

        /// <summary>
        /// number of epochs for training. 
        /// </summary>
        public static uint MaxEpochs = 1;

        private static readonly int[] imageDim = {32, 32, 3};
        private static readonly int numClasses = 10;

        /// <summary>
        /// Train and evaluate an image classifier with CIFAR-10 data. 
        /// The classification model is saved after training.
        /// For repeated runs, the caller may choose whether to retrain a model or 
        /// just validate an existing one.
        /// </summary>
        /// <param name="device">CPU or GPU device to run</param>
        /// <param name="forceRetrain">whether to override an existing model.
        /// if true, any existing model will be overridden and the new one evaluated. 
        /// if false and there is an existing model, the existing model is evaluated.</param>
        public static void TrainAndEvaluate(DeviceDescriptor device, bool forceRetrain)
        {
            string modelFile = "Cifar10Rest.model";

            // If a model already exists and not set to force retrain, validate the model and return.
            if (File.Exists(modelFile) && !forceRetrain)
            {
                ValidateModel(device, modelFile);
                return;
            }

            // prepare training data
            var minibatchSource = CreateMinibatchSource(Path.Combine(CifarDataFolder, "train_map.txt"),
                Path.Combine(CifarDataFolder, "CIFAR-10_mean.xml"), imageDim, numClasses, MaxEpochs);
            var imageStreamInfo = minibatchSource.StreamInfo("features");
            var labelStreamInfo = minibatchSource.StreamInfo("labels");

            // build a model
            var imageInput = CNTKLib.InputVariable(imageDim, imageStreamInfo.m_elementType, "Images");
            var labelsVar = CNTKLib.InputVariable(new int[] { numClasses }, labelStreamInfo.m_elementType, "Labels");
            var classifierOutput = ResNetClassifier(imageInput, numClasses, device, "classifierOutput");

            // prepare for training
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labelsVar, "lossFunction");
            var prediction = CNTKLib.ClassificationError(classifierOutput, labelsVar, 5, "predictionError");

            var learningRatePerSample = new TrainingParameterScheduleDouble(0.0078125, 1);
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction,
                new List<Learner> { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) });

            uint minibatchSize = 64;
            int outputFrequencyInMinibatches = 20, miniBatchCount = 0;

            // Feed data to the trainer for number of epochs. 
            while (true)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);

                // Stop training once max epochs is reached.
                if (minibatchData.empty())
                {
                    break;
                }

                trainer.TrainMinibatch(new Dictionary<Variable, MinibatchData>()
                        { { imageInput, minibatchData[imageStreamInfo] }, { labelsVar, minibatchData[labelStreamInfo] } }, device);
                TestHelper.PrintTrainingProgress(trainer, miniBatchCount++, outputFrequencyInMinibatches);
            }

            // save the model
            var imageClassifier = Function.Combine(new List<Variable>() { trainingLoss, prediction, classifierOutput }, "ImageClassifier");
            imageClassifier.Save(modelFile);

            // validate the model
            ValidateModel(device, modelFile);
        }

        private static void ValidateModel(DeviceDescriptor device, string modelFile)
        {
            MinibatchSource testMinibatchSource = CreateMinibatchSource(
                Path.Combine(CifarDataFolder, "test_map.txt"),
                Path.Combine(CifarDataFolder, "CIFAR-10_mean.xml"), imageDim, numClasses, 1);
            TestHelper.ValidateModelWithMinibatchSource(modelFile, testMinibatchSource,
                imageDim, numClasses, "features", "labels", "classifierOutput", device);
        }

        private static Function ConvBatchNormalizationReLULayer(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, int hStride, int vStride, 
            double wScale, double bValue, double scValue, int bnTimeConst, bool spatial, DeviceDescriptor device)
        {
            var convBNFunction = ConvBatchNormalizationLayer(input, outFeatureMapCount, kernelWidth, kernelHeight, hStride, vStride, wScale, bValue, scValue, bnTimeConst, spatial, device);
            return CNTKLib.ReLU(convBNFunction);
        }

        private static Function ConvBatchNormalizationLayer(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, int hStride, int vStride,
            double wScale, double bValue, double scValue, int bnTimeConst, bool spatial, DeviceDescriptor device)
        {
            int numInputChannels = input.Shape[input.Shape.Rank - 1];

            var convParams = new Parameter(new int[] { kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount },
                DataType.Float, CNTKLib.GlorotUniformInitializer(wScale, -1, 2), device);
            var convFunction = CNTKLib.Convolution(convParams, input, new int[] { hStride, vStride, numInputChannels });

            var biasParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)bValue, device, "");
            var scaleParams = new Parameter(new int[] { NDShape.InferredDimension }, (float)scValue, device, "");
            var runningMean = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, device);
            var runningInvStd = new Constant(new int[] { NDShape.InferredDimension }, 0.0f, device);
            var runningCount = Constant.Scalar(0.0f, device);
            return CNTKLib.BatchNormalization(convFunction, scaleParams, biasParams, runningMean, runningInvStd, runningCount,
                spatial, (double)bnTimeConst, 0.0, 1e-5 /* epsilon */);
        }

        private static Function ResNetNode(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, double wScale, double bValue,
            double scValue, int bnTimeConst, bool spatial, DeviceDescriptor device)
        {
            var c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
            var c2 = ConvBatchNormalizationLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
            var p = CNTKLib.Plus(c2, input);
            return CNTKLib.ReLU(p);
        }

        private static Function ResNetNodeInc(Variable input, int outFeatureMapCount, int kernelWidth, int kernelHeight, double wScale, double bValue,
            double scValue, int bnTimeConst, bool spatial, Variable wProj, DeviceDescriptor device)
        {
            var c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
            var c2 = ConvBatchNormalizationLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

            var cProj = ProjectLayer(wProj, input, 2, 2, bValue, scValue, bnTimeConst, device);

            var p = CNTKLib.Plus(c2, cProj);
            return CNTKLib.ReLU(p);
        }

        private static Function ProjectLayer(Variable wProj, Variable input, int hStride, int vStride, double bValue, double scValue, int bnTimeConst,
            DeviceDescriptor device)
        {
            int outFeatureMapCount = wProj.Shape[0];
            var b = new Parameter(new int[] { outFeatureMapCount }, (float)bValue, device, "");
            var sc = new Parameter(new int[] { outFeatureMapCount }, (float)scValue, device, "");
            var m = new Constant(new int[] { outFeatureMapCount }, 0.0f, device);
            var v = new Constant(new int[] { outFeatureMapCount }, 0.0f, device);

            var n = Constant.Scalar(0.0f, device);

            int numInputChannels = input.Shape[input.Shape.Rank - 1];

            var c = CNTKLib.Convolution(wProj, input, new int[] { hStride, vStride, numInputChannels }, new bool[] { true }, new bool[] { false });
            return CNTKLib.BatchNormalization(c, sc, b, m, v, n, true /*spatial*/, (double)bnTimeConst, 0, 1e-5, false);
        }

        private static Constant GetProjectionMap(int outputDim, int inputDim, DeviceDescriptor device)
        {
            if (inputDim > outputDim)
                throw new Exception("Can only project from lower to higher dimensionality");

            float[] projectionMapValues = new float[inputDim * outputDim];
            for (int i = 0; i < inputDim * outputDim; i++)
                projectionMapValues[i] = 0;
            for (int i = 0; i < inputDim; ++i)
                projectionMapValues[(i * (int)inputDim) + i] = 1.0f;

            var projectionMap = new NDArrayView(DataType.Float, new int[] { 1, 1, inputDim, outputDim }, device);
            projectionMap.CopyFrom(new NDArrayView(new int[] { 1, 1, inputDim, outputDim }, projectionMapValues, (uint)projectionMapValues.Count(), device));

            return new Constant(projectionMap);
        }

        /// <summary>
        /// Build a Resnet for image classification. 
        /// https://arxiv.org/abs/1512.03385
        /// </summary>
        /// <param name="input">input variable for image data</param>
        /// <param name="numOutputClasses">number of output classes</param>
        /// <param name="device">CPU or GPU device to run</param>
        /// <param name="outputName">name of the classifier</param>
        /// <returns></returns>
        private static Function ResNetClassifier(Variable input, int numOutputClasses, DeviceDescriptor device, string outputName)
        {
            double convWScale = 7.07;
            double convBValue = 0;

            double fc1WScale = 0.4;
            double fc1BValue = 0;

            double scValue = 1;
            int bnTimeConst = 4096;

            int kernelWidth = 3;
            int kernelHeight = 3;

            double conv1WScale = 0.26;
            int cMap1 = 16;
            var conv1 = ConvBatchNormalizationReLULayer(input, cMap1, kernelWidth, kernelHeight, 1, 1, conv1WScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

            var rn1_1 = ResNetNode(conv1, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn1_2 = ResNetNode(rn1_1, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);
            var rn1_3 = ResNetNode(rn1_2, cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

            int cMap2 = 32;
            var rn2_1_wProj = GetProjectionMap(cMap2, cMap1, device);
            var rn2_1 = ResNetNodeInc(rn1_3, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn2_1_wProj, device);
            var rn2_2 = ResNetNode(rn2_1, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn2_3 = ResNetNode(rn2_2, cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, device);

            int cMap3 = 64;
            var rn3_1_wProj = GetProjectionMap(cMap3, cMap2, device);
            var rn3_1 = ResNetNodeInc(rn2_3, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true /*spatial*/, rn3_1_wProj, device);
            var rn3_2 = ResNetNode(rn3_1, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);
            var rn3_3 = ResNetNode(rn3_2, cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false /*spatial*/, device);

            // Global average pooling
            int poolW = 8;
            int poolH = 8;
            int poolhStride = 1;
            int poolvStride = 1;
            var pool = CNTKLib.Pooling(rn3_3, PoolingType.Average,
                new int[] { poolW, poolH, 1 }, new int[] { poolhStride, poolvStride, 1 });

            // Output DNN layer
            var outTimesParams = new Parameter(new int[] { numOutputClasses, 1, 1, cMap3 }, DataType.Float,
                CNTKLib.GlorotUniformInitializer(fc1WScale, 1, 0), device);
            var outBiasParams = new Parameter(new int[] { numOutputClasses }, (float)fc1BValue, device, "");

            return CNTKLib.Plus(CNTKLib.Times(outTimesParams, pool), outBiasParams, outputName);
        }

        private static MinibatchSource CreateMinibatchSource(string mapFilePath, string meanFilePath,
            int[] imageDims, int numClasses, uint maxSweeps)
        {
            List<CNTKDictionary> transforms = new List<CNTKDictionary>{
                CNTKLib.ReaderCrop("RandomSide",
                    new Tuple<int, int>(0, 0),
                    new Tuple<float, float>(0.8f, 1.0f),
                    new Tuple<float, float>(0.0f, 0.0f),
                    new Tuple<float, float>(1.0f, 1.0f),
                    "uniRatio"),
                CNTKLib.ReaderScale(imageDims[0], imageDims[1], imageDims[2]),
                CNTKLib.ReaderMean(meanFilePath)
            };

            var deserializerConfiguration = CNTKLib.ImageDeserializer(mapFilePath,
                "labels", (uint)numClasses,
                "features",
                transforms);

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfiguration })
            {
                MaxSweeps = maxSweeps
            };

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }
    }
}
