using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    public class LSTMSequenceClassifierTest
    {
        static Function Stabilize<ElementType>(Variable x, DeviceDescriptor device)
        {
            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log( 
                    Constant.Scalar(f.DataType, 1.0) +  
                    CNTKLib.Exp(CNTKLib.ElementTimes(f, new Parameter(new NDShape(), f.DataType, 0.99537863 /* 1/f*ln (e^f-1) */, device)))));
            return CNTKLib.ElementTimes(beta, x);
        }

        static Tuple<Function, Function> LSTMPCellWithSelfStabilization<ElementType>( 
            Variable input, Variable prevOutput, Variable prevCellState, DeviceDescriptor device)
        {
            int outputDim = prevOutput.Shape[0];
            int cellDim = prevCellState.Shape[0];

            bool isFloatType = typeof(ElementType).Equals(typeof(float));
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            // new Parameter(new NDShape(new uint[] { 1 }), (ElementType)(object)0.0, device, "");
            // TODO, how to use ElementType?
            Func<int, Parameter> createBiasParam;
            if (isFloatType)
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01f, device, "");
            else
                createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01, device, "");

            uint seed2 = 1;
            Func<int, Parameter> createProjectionParam = (oDim) => new Parameter(new int[] { oDim, NDShape.InferredDimension },
                    dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<int, Parameter> createDiagWeightParam = (dim) =>
                new Parameter(new int[] { dim }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Function stabilizedPrevOutput = Stabilize<ElementType>(prevOutput, device);
            Function stabilizedPrevCellState = Stabilize<ElementType>(prevCellState, device);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + (createProjectionParam(cellDim) * input);

            // Input gate
            Function it =
                CNTKLib.Sigmoid(
                    (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +  
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bit = CNTKLib.ElementTimes(
                it,
                CNTKLib.Tanh(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)));

            // Forget-me-not gate
            Function ft = CNTKLib.Sigmoid(
                (Variable)(
                        projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) +
                        CNTKLib.ElementTimes(createDiagWeightParam(cellDim), stabilizedPrevCellState));
            Function bft = CNTKLib.ElementTimes(ft, prevCellState);

            Function ct = (Variable)bft + bit;

            // Output gate
            Function ot = CNTKLib.Sigmoid( 
                (Variable)(projectInput() + (createProjectionParam(cellDim) * stabilizedPrevOutput)) + 
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), Stabilize<ElementType>(ct, device)));
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            Function c = ct;
            Function h = (outputDim != cellDim) ? (createProjectionParam(outputDim) * Stabilize<ElementType>(ht, device)) : ht;

            return new Tuple<Function, Function>(h, c);
        }


        static Tuple<Function, Function> LSTMPComponentWithSelfStabilization<ElementType>(Variable input,
            NDShape outputShape, NDShape cellShape,
            Func<Variable, Function> recurrenceHookH,
            Func<Variable, Function> recurrenceHookC,
            DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var LSTMCell = LSTMPCellWithSelfStabilization<ElementType>(input, dh, dc, device);
            var actualDh = recurrenceHookH(LSTMCell.Item1);
            var actualDc = recurrenceHookC(LSTMCell.Item2);

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            (LSTMCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });

            return new Tuple<Function, Function>(LSTMCell.Item1, LSTMCell.Item2);
        }



        private static Function Embedding(Variable input, int embeddingDim, DeviceDescriptor device)
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];
            var embeddingParameters = new Parameter(new int[] { embeddingDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
            return CNTKLib.Times(embeddingParameters, input);
        }

        static Function LSTMSequenceClassifierNet(Variable input, int numOutputClasses, int embeddingDim, int LSTMDim, int cellDim, DeviceDescriptor device, 
            string outputName)
        {
            Function embeddingFunction = Embedding(input, embeddingDim, device);
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);
            Function LSTMFunction = LSTMPComponentWithSelfStabilization<float>(
                embeddingFunction,
                new int[] { LSTMDim },
                new int[] { cellDim },
                pastValueRecurrenceHook,
                pastValueRecurrenceHook,
                device).Item1;
            Function thoughtVectorFunction = CNTKLib.Last(LSTMFunction);

            return TestHelper.FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
        }

        public static void TrainLSTMSequenceClassifier(DeviceDescriptor device, bool useSparseLabels, bool testSaveAndReLoad)
        {
            const int inputDim = 2000;
            const int cellDim = 25;
            const int hiddenDim = 25;
            const int embeddingDim = 50;
            const int numOutputClasses = 5;

            var featuresName = "features";
            var features = CNTKLib.InputVariable(new int[] { inputDim }, true /*isSparse*/, DataType.Float, featuresName);
            var classifierOutput = LSTMSequenceClassifierNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, "classifierOutput");

            var labelsName = "labels";
            var labels = Variable.InputVariable(new int[] { numOutputClasses }, useSparseLabels, DataType.Float, labelsName,
                new List<Axis>() { Axis.DefaultBatchAxis() });
            Function trainingLoss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labels, "lossFunction");
            Function prediction = CNTKLib.ClassificationError(classifierOutput, labels, "classificationError");

            if (testSaveAndReLoad)
            {
                Variable classifierOutputVar = classifierOutput;
                Variable trainingLossVar = trainingLoss;
                Variable predictionVar = prediction;
                var oneHiddenLayerClassifier = Function.Combine(new List<Variable>() { trainingLossVar, predictionVar, classifierOutputVar }, "classifierModel");
                TestHelper.SaveAndReloadModel(ref oneHiddenLayerClassifier, new List<Variable>() { features, labels, trainingLossVar, predictionVar, classifierOutputVar }, device);

                // NOT NEEDED
                classifierOutput = classifierOutputVar;
                trainingLoss = trainingLossVar;
                prediction = predictionVar;
            }

            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featuresName, inputDim, true, "x"), new StreamConfiguration(labelsName, numOutputClasses, false, "y") };

            // TODO:
            var minibatchSource = MinibatchSource.TextFormatMinibatchSource("Train.ctf", streamConfigurations,
                MinibatchSource.FullDataSweep, true);

            const uint minibatchSize = 200;

            var featureStreamInfo = minibatchSource.StreamInfo(featuresName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsName);

            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(
                0.0005, TrainingParameterScheduleDouble.UnitType.Sample);

            MomentumAsTimeConstantScheduleCS momentumTimeConstant = new MomentumAsTimeConstantScheduleCS(256);
            // CNTK.TrainingParameterScheduleDouble momentumTimeConstant = new CNTK.TrainingParameterScheduleDouble(256.0);

            IList<Learner> parameterLearners = new List<Learner>() {
                Learner.MomentumSGDLearner(classifierOutput.Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true)  };
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            int outputFrequencyInMinibatches = 1;
            for (int i = 0; true; i++)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                if (minibatchData.empty())
                    break;

                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { features, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };

                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }
        }
    }
}
