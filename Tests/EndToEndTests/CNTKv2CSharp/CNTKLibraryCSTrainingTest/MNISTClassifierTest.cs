using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    public class MNISTClassifierTest
    {
        public static void TrainMNISTClassifier(DeviceDescriptor device)
        {
            int inputDim = 784;
            int numOutputClasses = 10;
            int hiddenLayerDim = 200;

            var input = CNTKLib.InputVariable(new int[] { inputDim }, DataType.Float, "features");
            var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);

            Function toSigmoid = TestHelper.FullyConnectedLinearLayer(new Variable(scaledInput), hiddenLayerDim, device, "");
            var classifierOutput = CNTKLib.Sigmoid(new Variable(toSigmoid), "");


            var outputTimesParam = new Parameter(NDArrayView.RandomUniform<float>(
                new int[] { numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device));
            var outputBiasParam = new Parameter(NDArrayView.RandomUniform<float>(new int[] { numOutputClasses }, -0.05, 0.05, 1, device));
            classifierOutput = CNTKLib.Plus(outputBiasParam, new Variable(CNTKLib.Times(outputTimesParam, new Variable(classifierOutput))), "classifierOutput");

            var labels = CNTKLib.InputVariable(new int[] { numOutputClasses }, DataType.Float, "labels");
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError");

            // Test save and reload of model
            {
                Variable classifierOutputVar = new Variable(classifierOutput);
                Variable trainingLossVar = new Variable(trainingLoss);
                Variable predictionVar = new Variable(prediction);
                var combinedNet = Function.Combine(new List<Variable>() { trainingLossVar, predictionVar, classifierOutputVar }, "MNISTClassifier");
                TestHelper.SaveAndReloadModel(ref combinedNet, new List<Variable>() { input, labels, trainingLossVar, predictionVar, classifierOutputVar }, device);

                classifierOutput = classifierOutputVar.ToFunction();
                trainingLoss = trainingLossVar.ToFunction();
                prediction = predictionVar.ToFunction();
            }

            const uint minibatchSize = 64;
            const uint numSamplesPerSweep = 60000;
            const uint numSweepsToTrainWith = 2;
            const uint numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

            var featureStreamName = "features";
            var labelsStreamName = "labels";
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, inputDim), new StreamConfiguration(labelsStreamName, numOutputClasses) };

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource("Train-28x28_cntk_text.txt", streamConfigurations);

            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.003125, TrainingParameterScheduleDouble.UnitType.Sample);

            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            int outputFrequencyInMinibatches = 20;
            for (int i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { input, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }
        }

        static Function create_basic_model(DeviceDescriptor device, Variable input_var, int out_dims)
        {
            // just some baseline classification model
            // with convolutional layers

            // 1x64x64 -> 32x32x32

            int kernelWidth1 = 5, kernelHeight1 = 5, numInputChannels1 = 1, outFeatureMapCount1 = 32;
            int hStride1 = 2, vStride1 = 2;
            int poolingWindowWidth1 = 3, poolingWindowHeight1 = 3;
            BoolVector poolingPadding = new BoolVector();
            poolingPadding.Add(true);

            var convParams1 = new Parameter(new int[]{ kernelWidth1, kernelHeight1, numInputChannels1, outFeatureMapCount1 }, DataType.Float, 1.0F);
            Function convFunction1 = CNTKLib.Convolution(convParams1, input_var, new int[] { 1, 1, 1 });
            // TestHelper.PrintOutputDims(convFunction1, "convFunction1");
            Function pooling1 = CNTKLib.Pooling(convFunction1, PoolingType.Max, 
                new int[] { poolingWindowWidth1, poolingWindowHeight1 }, new int[] { hStride1, vStride1 }, poolingPadding);
            // TestHelper.PrintOutputDims(pooling1, "pooling1");

            // 32x32x32 -> 32x16x16
            int kernelWidth2 = 5, kernelHeight2 = 5, numInputChannels2 = 1, outFeatureMapCount2 = 32;
            int hStride2 = 2, vStride2 = 2;
            int poolingWindowWidth2 = 3, poolingWindowHeight2 = 3;

            var convParams2 = new Parameter(new int[] { kernelWidth2, kernelHeight2, numInputChannels2, outFeatureMapCount2 }, DataType.Float, 1.0F);
            Function convFunction2 = CNTKLib.Convolution(convParams2, pooling1, new int[] { 1, 1, outFeatureMapCount1 });
            // TestHelper.PrintOutputDims(convFunction2, "convFunction2");
            Function pooling2 = CNTKLib.Pooling(convFunction2, PoolingType.Max, 
                new int[] { poolingWindowWidth2, poolingWindowHeight2 }, new int[] { hStride2, vStride2 }, poolingPadding);
            // TestHelper.PrintOutputDims(pooling2, "pooling2");


            // 32x16x16 -> 64x8x8
            int kernelWidth3 = 5, kernelHeight3 = 5, numInputChannels3 = 1, outFeatureMapCount3 = 64;
            int hStride3 = 2, vStride3 = 2;
            int poolingWindowWidth3 = 3, poolingWindowHeight3 = 3;

            var convParams3 = new Parameter(new int[] { kernelWidth3, kernelHeight3, numInputChannels3, outFeatureMapCount3 }, DataType.Float, 1.0F);
            Function convFunction3 = CNTKLib.Convolution(convParams3, pooling2, new int[] { 1, 1, outFeatureMapCount2 });
            // TestHelper.PrintOutputDims(convFunction3, "convFunction3");
            Function pooling3 = CNTKLib.Pooling(convFunction3, PoolingType.Max, 
                new int[] { poolingWindowWidth3, poolingWindowHeight3 }, new int[] { hStride3, vStride3 }, poolingPadding);
            // TestHelper.PrintOutputDims(pooling3, "pooling3");

            Function toDense = CNTKLib.Reshape(pooling3, new int[] { 64 * 8 * 8 });

            // 64x8x8 -> 64 -> 3
            int hiddenLayerDim = 64;
            Function denseLayer1 = CNTKLib.Sigmoid(TestHelper.FullyConnectedLinearLayer(toDense, hiddenLayerDim, device));
            // TestHelper.PrintOutputDims(denseLayer1, "denseLayer1");
            Function denseLayer2 = CNTKLib.Sigmoid(TestHelper.FullyConnectedLinearLayer(denseLayer1, out_dims, device));
            // TestHelper.PrintOutputDims(denseLayer2, "denseLayer2");
            return denseLayer2;
        }

        public static void TrainMNISTClassifierWithConvolution(DeviceDescriptor device)
        {
            CNTKLib.SetTraceLevel(TraceLevel.Info);
            // https://github.com/Microsoft/CNTK/issues/1838
            // 1x64x64 -> 32x32x32
            //var foo = new Parameter(new NDShape(new uint[] { 2, 3 }), DataType.Float, 1.0f);
            //var bar = new Parameter(new NDShape(new uint[] { 3, (uint)NDShape.InferredDimension }), DataType.Float, 1.0f);
            //var baz = foo * bar;
            //var s = (baz.Output as Variable).Shape;

            // https://github.com/Microsoft/CNTK/issues/1838
            int num_channels = 1, image_height = 64, image_width = 64;
            Variable input_var = CNTKLib.InputVariable(new int[] { image_width, image_height, num_channels }, DataType.Float);

            int num_classes = 3;

            Function testConvolutionLayer = create_basic_model(device, input_var, num_classes);

            foreach (var n_test in new int[] { 230 }) //  1, 2, 4, 8, 16, 32, 64, 128, 256})
            {
                int size = (int)(n_test * num_channels * image_height * image_width);
                List<float> dataBuffer = new List<float>(size);
                for (int i = 0; i < size; i++)
                    dataBuffer.Add(0.3F);
                var inputVal = Value.CreateBatch(new int[] { image_height, image_width, num_channels }, dataBuffer, device);

                var inputs = new Dictionary<Variable, Value>();
                inputs.Add(input_var, inputVal);

                var outputs = testConvolutionLayer.Outputs.ToDictionary(v => v, v => (CNTK.Value)null);

                testConvolutionLayer.Evaluate(inputs, outputs, device);
            }
        }
    }
}
