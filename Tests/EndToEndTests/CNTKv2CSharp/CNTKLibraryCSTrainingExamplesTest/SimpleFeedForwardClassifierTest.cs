using System;
using System.Collections.Generic;
using CNTK;

namespace CNTKLibraryCSTrainingExamplesTest
{
    /// <summary> 
    /// The example shows
    /// - how to prepare training data with minibatch source.
    /// - how to build a data classify model.
    /// - how to save and reload the model.
    /// - how to train the model.
    /// </summary>
    /// <param name="device">Specify on which device to run the evaluation.</param>
    public class SimpleFeedForwardClassifierTest
    {
        internal static void TrainSimpleFeedForwardClassifier(DeviceDescriptor device)
        {
            int inputDim = 2;
            int numOutputClasses = 2;
            int hiddenLayerDim = 50;
            int numHiddenLayers = 2;

            int minibatchSize = 50;
            int numSamplesPerSweep = 10000;
            int numSweepsToTrainWith = 2;
            int numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;

            var featureStreamName = "features";
            var labelsStreamName = "labels";
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, inputDim), new StreamConfiguration(labelsStreamName, numOutputClasses) };

            var minibatchSource = MinibatchSource.TextFormatMinibatchSource("SimpleDataTrain_cntk_text.txt", 
                streamConfigurations, MinibatchSource.FullDataSweep, true, MinibatchSource.DefaultRandomizationWindowInChunks);
            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            IDictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>> inputMeansAndInvStdDevs =
                new Dictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>>
                { { featureStreamInfo, new Tuple<NDArrayView, NDArrayView>(null, null) } };
            MinibatchSource.ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, inputMeansAndInvStdDevs, device);

            var input = CNTKLib.InputVariable(new int[] { inputDim }, DataType.Float, "features");
            var normalizedinput = CNTKLib.PerDimMeanVarianceNormalize(input, 
                inputMeansAndInvStdDevs[featureStreamInfo].Item1, inputMeansAndInvStdDevs[featureStreamInfo].Item2);
            Function fullyConnected = TestHelper.FullyConnectedLinearLayer(normalizedinput, hiddenLayerDim, device, "");
            var classifierOutput = CNTKLib.Sigmoid(fullyConnected, "");

            for (int i = 1; i < numHiddenLayers; ++i)
            {
                fullyConnected = TestHelper.FullyConnectedLinearLayer(classifierOutput, hiddenLayerDim, device, "");
                classifierOutput = CNTKLib.Sigmoid(fullyConnected, "");
            }

            var outputTimesParam = new Parameter(NDArrayView.RandomUniform<float>(
                new int[] { numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device));
            var outputBiasParam = new Parameter(NDArrayView.RandomUniform<float>(
                new int[] { numOutputClasses }, -0.05, 0.05, 1, device));
            classifierOutput = CNTKLib.Plus(outputBiasParam, outputTimesParam * classifierOutput, "classifierOutput");

            var labels = CNTKLib.InputVariable(new int[] { numOutputClasses }, DataType.Float, "labels");
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labels, "lossFunction"); ;
            var prediction = CNTKLib.ClassificationError(classifierOutput, labels, "classificationError");

            // Test save and reload of model
            {
                Variable classifierOutputVar = classifierOutput;
                Variable trainingLossVar = trainingLoss;
                Variable predictionVar = prediction;
                var combinedNet = CNTKLib.Combine(new List<Variable>() { trainingLoss, prediction, classifierOutput },
                    "feedForwardClassifier");
                TestHelper.SaveAndReloadModel(ref combinedNet, 
                    new List<Variable>() { input, labels, trainingLossVar, predictionVar, classifierOutputVar }, device);

                classifierOutput = classifierOutputVar;
                trainingLoss = trainingLossVar;
                prediction = predictionVar;
            }


            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.02, TrainingParameterScheduleDouble.UnitType.Sample);

            streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration("features", inputDim), new StreamConfiguration("labels", numOutputClasses) };
            minibatchSource = MinibatchSource.TextFormatMinibatchSource("SimpleDataTrain_cntk_text.txt", streamConfigurations);

            IList<Learner> parameterLearners = 
                new List<Learner>() { CNTKLib.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            int outputFrequencyInMinibatches = 20;
            int trainingCheckpointFrequency = 100;
            for (int i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch((uint)minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { input, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

                if ((i % trainingCheckpointFrequency) == (trainingCheckpointFrequency - 1))
                {
                    string ckpName = "feedForward.net";
                    trainer.SaveCheckpoint(ckpName);
                    trainer.RestoreFromCheckpoint(ckpName);
                }
            }
        }
    }
}
