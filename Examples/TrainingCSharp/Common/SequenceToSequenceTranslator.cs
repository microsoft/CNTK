using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CNTK.CSTrainingExamples
{
    /// <summary>
    /// This class shows how to build a recurrent neural network model from ground up and train the model. 
    /// </summary>
    public class SequenceToSequenceTranslator
    {
        interface ISequenceDataSource
        {
            bool GetNextMinibatch(int minibatchSize, DeviceDescriptor device, out Value features, out Value labels);
            void PrintTranslation(Value featureValue, Value lavelValue, Variable rawLabels, Variable labels, DeviceDescriptor device);
        }

        public class SequenceDataSource : ISequenceDataSource
        {
            int inputVocabDim;
            int labelVocabDim;
            Dictionary<int, string> i2w;
            Dictionary<string, int> w2i;
            IList<Tuple<IList<string>, IList<string>>> graphemeToPhonemes;
            int sampleIndex;

            public SequenceDataSource(int inputVocabDim, int labelVocabDim, string dataFilepath, int epochSize)
            {
                this.inputVocabDim = inputVocabDim;
                this.labelVocabDim = labelVocabDim;
                LoadGraphemeToPhonemeData(dataFilepath, out i2w, out w2i, out graphemeToPhonemes);
                RendomShaffle(graphemeToPhonemes);
                graphemeToPhonemes = graphemeToPhonemes.Take(epochSize).ToList();
                sampleIndex = 0;
            }

            public void PrintTranslation(Value featureValue, Value lavelValue, Variable rawLabels, Variable labels, DeviceDescriptor device)
            {
                IList<NDArrayView> featureViews = featureValue.UnpackVariableValue(rawLabels, device);
                //IList<NDArrayView> labelViews = lavelValue.UnpackVariableValue(labels, device);
                //if (featureViews.Count != labelViews.Count)
                //{
                //    throw new Exception("Translation shall produce equal amount of output sequences as input.");
                //}

                IList<IList<int>> featureOnehotSeqs = featureValue.GetOneHotData(rawLabels);

                IList<IList<float>> labelDenseSeqs = lavelValue.GetDenseData<float>(labels);

                for (int index = 0; index < featureViews.Count; index++)
                {
                    List<int> featureOnehotSeq = featureOnehotSeqs[index].ToList();
                    List<string> graphemes = new List<string>();
                    featureOnehotSeq.ForEach(f => graphemes.Add(i2w[f]));
                    Print(graphemes);
                    List<float> labelSoftmaxSeq = labelDenseSeqs[index].ToList();

                    List<string> phonemes = new List<string>();
                    int phonemeCount = labelSoftmaxSeq.Count / this.labelVocabDim;
                    for (int i = 0; i < phonemeCount; i++)
                    {
                        List<float> labelSoftmaxList = labelSoftmaxSeq.Skip(i * this.labelVocabDim).Take(this.labelVocabDim).ToList();
                        int actualLabel = labelSoftmaxList.IndexOf(labelSoftmaxList.Max());
                        phonemes.Add(i2w[actualLabel]);
                    }

                    Print(phonemes);
                }
            }

            IList<string> ConvertViewToSequence(NDArrayView view, Variable variable, Dictionary<int, string> i2w)
            { 
                Value seqValue = new Value(view);
                if (seqValue.IsSparse)
                {
                    int featuresequenceLength;
                    IList<int> featurecolStarts;
                    IList<int> featurerowIndices;
                    IList<float> featurenonZeroValues;
                    int featurenumNonZeroValues;
                    seqValue.GetSparseData<float>(variable, out featuresequenceLength, out featurecolStarts,
                                        out featurerowIndices, out featurenonZeroValues, out featurenumNonZeroValues);

                    List<string> sequence = new List<string>();
                    foreach (int index in featurerowIndices)
                    {
                        sequence.Add(i2w[index]);
                    }
                    return sequence;
                }
                else
                {
                    return null;
                    //IList<IList<int>> featurerowIndices = seqValue.GetOneHotData(variable);
                    //List<string> sequence = new List<string>();
                    //foreach (int index in featurerowIndices)
                    //{
                    //    sequence.Add(i2w[index]);
                    //}
                    //return sequence;
                }
            }

            public bool GetNextMinibatch(int minibatchSize, DeviceDescriptor device, out Value featureValue, out Value labelValue)
            {
                featureValue = null;
                labelValue = null;
                if (graphemeToPhonemes.Count <= sampleIndex)
                {
                    return false;
                }

                int graphemeCount = 0;
                IList<IEnumerable<int>> batchOfGraphemeSequences = new List<IEnumerable<int>>();
                IList<IEnumerable<int>> batchOfPhonemeSequences = new List<IEnumerable<int>>();
                while (sampleIndex < graphemeToPhonemes.Count)
                {
                    IList<string> graphemes = graphemeToPhonemes[sampleIndex].Item1;
                    if (graphemeCount + graphemes.Count > minibatchSize)
                    {
                        break;
                    }
                    graphemeCount += graphemes.Count;

                    IList<string> phonemes = graphemeToPhonemes[sampleIndex].Item2;

                    List<int> graphemeSequence = new List<int>();
                    graphemes.ToList().ForEach(g => graphemeSequence.Add(w2i[g]));
                    batchOfGraphemeSequences.Add(graphemeSequence);

                    List<int> phonemeSequence = new List<int>();
                    phonemes.ToList().ForEach(p => phonemeSequence.Add(w2i[p]));
                    batchOfPhonemeSequences.Add(phonemeSequence);

                    sampleIndex++;
                }

                featureValue = Value.CreateBatchOfSequences<float>(inputVocabDim, batchOfGraphemeSequences, device);
                labelValue = Value.CreateBatchOfSequences<float>(labelVocabDim, batchOfPhonemeSequences, device);

                //SequenceToSequenceTranslator.QuerySparseData(featureValue, inputVocabDim, device);
                //SequenceToSequenceTranslator.QuerySparseData(labelValue, labelVocabDim, device);
                return true;
            }

            void RendomShaffle(IList<Tuple<IList<string>, IList<string>>> graphemeToPhonemes)
            { 
                // randomize 
                int n = graphemeToPhonemes.Count;
                Random random = new Random(0);
                while (n > 1)
                {
                    n--;
                    int k = random.Next(n + 1);
                    var value = graphemeToPhonemes[k];
                    graphemeToPhonemes[k] = graphemeToPhonemes[n];
                    graphemeToPhonemes[n] = value;
                }
            }

            static void LoadGraphemeToPhonemeData(string filepath, 
                out Dictionary<int, string> i2w, out Dictionary<string, int> w2i, 
                out IList<Tuple<IList<string>, IList<string>>> graphemeToPhonemes)
            {
                string sStart = "<s>";
                string sEnd = "</s>";
                string middleBreak = "<s/>";
                string[] lines = System.IO.File.ReadAllLines(filepath);

                HashSet<string> dict = new HashSet<string>();
                dict.Add("'");
                dict.Add(sEnd);
                dict.Add(middleBreak);
                dict.Add(sStart);

                graphemeToPhonemes = new List<Tuple<IList<string>, IList<string>>>();
                foreach (var line in lines)
                {
                    List<string> graphemePhonemes = line.Split(' ').ToList();

                    int midIndex = graphemePhonemes.IndexOf(middleBreak);
                    List<string> graphemes = graphemePhonemes.Take(midIndex + 1).ToList();
                    graphemes.Except(dict).ToList().ForEach(g => dict.Add(g));
                    graphemes[graphemes.Count() - 1] = sEnd;

                    List<string> phonemes = graphemePhonemes.Skip(midIndex).ToList();
                    phonemes[0] = sStart;
                    phonemes.Except(dict).ToList().ForEach(p => dict.Add(p));

                    graphemeToPhonemes.Add(new Tuple<IList<string>, IList<string>>(graphemes, phonemes));
                    // Print(graphemes);
                    // Print(phonemes);
                }

                i2w = new Dictionary<int, string>();
                w2i = new Dictionary<string, int>();
                foreach (string e in dict)
                {
                    i2w.Add(i2w.Count, e);
                    w2i.Add(e, w2i.Count);
                }
            }

            static void Print(IEnumerable<string> phonemes)
            {
                Console.WriteLine(String.Join("-", phonemes));
            }
        }

        public class SequenceDataSourceCTF : ISequenceDataSource
        {
            string featureStreamName = "rawInput";
            string labelStreamName = "rawLabels";
            MinibatchSource minibatchSource;
            StreamInformation rawInputStreamInfo;
            StreamInformation rawLabelsStreamInfo;

            public SequenceDataSourceCTF(int inputVocabDim, int labelVocabDim, string dataFilepath, int epochSize)
            {
                StreamConfiguration[] streamConfigurations = new StreamConfiguration[] {
                    new StreamConfiguration(featureStreamName, inputVocabDim, true, "S0"),
                    new StreamConfiguration(labelStreamName, labelVocabDim, true, "S1")
                };

                minibatchSource = MinibatchSource.TextFormatMinibatchSource(dataFilepath, streamConfigurations, (ulong)epochSize);

                rawInputStreamInfo = minibatchSource.StreamInfo(featureStreamName);
                rawLabelsStreamInfo = minibatchSource.StreamInfo(labelStreamName);
            }

            public void PrintTranslation(Value featureValue, Value lavelValue, Variable rawLabels, Variable labels, DeviceDescriptor device)
            {
                IList<IList<float>> labelDenseSeqs = lavelValue.GetDenseData<float>(labels);
                throw new NotImplementedException();
            }

            public bool GetNextMinibatch(int minibatchSize, DeviceDescriptor device, out Value features, out Value labels)
            {

                var minibatchData = minibatchSource.GetNextMinibatch((uint)minibatchSize, device);
                if (!minibatchData.empty())
                {
                    features = minibatchData[rawInputStreamInfo].data;
                    labels = minibatchData[rawLabelsStreamInfo].data;
                    return true;
                }
                else
                {
                    features = null;
                    labels = null;
                    return false;
                }
            }
        }

        public static void QuerySparseData(Value value, int dim, DeviceDescriptor device)
        {
            IList<Axis> inputDynamicAxes = new List<Axis> { new Axis("inputAxis"), Axis.DefaultBatchAxis() };
            var rawInput = Variable.InputVariable(new int[] { dim }, DataType.Float, "rawInput", inputDynamicAxes, true);

            IList<NDArrayView> ndArrayViews = value.UnpackVariableValue(rawInput, device);
            foreach (NDArrayView ndArrayView in ndArrayViews)
            {
                int featuresequenceLength;
                IList<int> featurecolStarts;
                IList<int> featurerowIndices;
                IList<float> featurenonZeroValues;
                int featurenumNonZeroValues;
                StorageFormat storageFormat = ndArrayView.StorageFormat;
                Value seqValue = new Value(ndArrayView);
                seqValue.GetSparseData<float>(rawInput, out featuresequenceLength, out featurecolStarts,
                                    out featurerowIndices, out featurenonZeroValues, out featurenumNonZeroValues);

                featurecolStarts = featurecolStarts.ToList();
                featurerowIndices = featurerowIndices.ToList();
                featurenonZeroValues = featurenonZeroValues.ToList();
            }
        }

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

        /// <summary>
        /// Build a one direction recurrent neural network (RNN) with long-short-term-memory (LSTM) cells.
        /// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        /// </summary>
        /// <param name="input">the input variable</param>
        /// <param name="numOutputClasses">number of output classes</param>
        /// <param name="embeddingDim">dimension of the embedding layer</param>
        /// <param name="LSTMDim">LSTM output dimension</param>
        /// <param name="cellDim">cell dimension</param>
        /// <param name="device">CPU or GPU device to run the model</param>
        /// <param name="outputName">name of the model output</param>
        /// <returns>the RNN model</returns>
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
            Function thoughtVectorFunction = CNTKLib.SequenceLast(LSTMFunction);

            return TestHelper.FullyConnectedLinearLayer(thoughtVectorFunction, numOutputClasses, device, outputName);
        }

        /// <summary>
        /// Execution folder is: CNTK/x64/BuildFolder
        /// Data folder is: CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data
        /// </summary>
        public static string DataFolder = "../../Examples/SequenceToSequence/CMUDict/Data";

        /// <summary>
        /// Build and train a RNN model.
        /// </summary>
        /// <param name="device">CPU or GPU device to train and run the model</param>
        public static void TrainAndEvaluate(DeviceDescriptor device, bool useCTFFormat, bool forceRetrain)
        {
            bool useSparseInputs = true;
            int inputVocabDim = 69;
            int labelVocabDim = 69;

            int hiddenDim = 512;
            int numLayers = 2;

            int embeddingDim = 300;
            int inputEmbeddingDim = Math.Min(inputVocabDim, embeddingDim);
            int labelEmbeddingDim = Math.Min(labelVocabDim, embeddingDim);

            /* Inputs */
            IList<Axis> inputDynamicAxes = new List<Axis> { new Axis("inputAxis"), Axis.DefaultBatchAxis() };

            var rawInput = Variable.InputVariable(new int[] { inputVocabDim }, DataType.Float, "rawInput", inputDynamicAxes, useSparseInputs);

            IList<Axis> labelDynamicAxes = new List<Axis> { new Axis("labelAxis"), Axis.DefaultBatchAxis() };
            var rawLabels = Variable.InputVariable(new int[] { labelVocabDim }, DataType.Float, "rawLabels", labelDynamicAxes, useSparseInputs);

            Function inputSequence = CNTKLib.Alias(rawInput, "inputSequence");

            // Drop the sentence start token from the label, for decoder training
            var labelSequence = CNTKLib.SequenceSlice(rawLabels, 1, 0, "labelSequenceWithStartTrimmed");
            Function decoderHistoryHook;
            Function seqToSeqModel = BuildSeqToSeqModel(device, useSparseInputs, inputVocabDim, labelVocabDim, hiddenDim, numLayers, inputEmbeddingDim, labelEmbeddingDim, rawLabels, inputSequence, labelSequence, out decoderHistoryHook);

            var ce = CNTKLib.CrossEntropyWithSoftmax(seqToSeqModel, labelSequence, "lossFunction");
            var errs = CNTKLib.ClassificationError(seqToSeqModel, labelSequence, "classificationError");

            // Decoder history for greedy decoding
            var decoderHistoryFromOutput = CNTKLib.Hardmax(seqToSeqModel);
            var decodingFunction = decoderHistoryFromOutput.Clone(ParameterCloningMethod.Share, new Dictionary<Variable, Variable> { { decoderHistoryHook, decoderHistoryFromOutput } });
            decodingFunction = Function.Combine(new List<Variable> { decodingFunction.RootFunction.Arguments[0] });

            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.007, 1);
            TrainingParameterScheduleDouble momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(1100);

            AdditionalLearningOptions additionalOptions = new AdditionalLearningOptions();
            additionalOptions.gradientClippingThresholdPerSample = 2.3;
            additionalOptions.gradientClippingWithTruncation = true;

            var trainer = Trainer.CreateTrainer(seqToSeqModel, ce, errs, new List<Learner>() {
                Learner.MomentumSGDLearner(seqToSeqModel.Parameters(), learningRatePerSample, momentumTimeConstant, /*unitGainMomentum = */true, additionalOptions) });

            int epochSize = 100000;
            ISequenceDataSource sequenceDataSource;
            if (useCTFFormat)
                sequenceDataSource = new SequenceDataSourceCTF(inputVocabDim, labelVocabDim, 
                    Path.Combine(DataFolder, "cmudict-0.7b.train-dev-20-21.ctf"), epochSize);
            else
                sequenceDataSource = new SequenceDataSource(inputVocabDim, labelVocabDim, 
                    Path.Combine(DataFolder, "cmudict-0.7b.train-dev-20-21.txt"), epochSize);

            int outputFrequencyInMinibatches = 1;
            int minibatchSize1 = 72 * 10;
            int minibatchSize2 = 144 * 10;
            int numMinbatchesToChangeMinibatchSizeAfter = 30;
            // string modelFile = "seq2seq.model";
            int decodingFrequency = 10;
            for (int i = 0; true; i++)
            {
                var minibatchSize = (i >= numMinbatchesToChangeMinibatchSizeAfter) ? minibatchSize2 : minibatchSize1;
                Value featureValue, labelValue;
                if (!sequenceDataSource.GetNextMinibatch(minibatchSize, device, out featureValue, out labelValue))
                    break;

                trainer.TrainMinibatch(
                    new Dictionary<Variable, Value>
                    { { rawInput, featureValue }, { rawLabels, labelValue } },
                    false,
                    device);

                if (trainer.PreviousMinibatchLossAverage() < 0 || trainer.PreviousMinibatchEvaluationAverage() < 0)
                    throw new Exception("SequenceToSequence: Invalid (-ve or nan) loss or evaluation metric encountered in training of the SequenceToSequence model.");

                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);

                if ((i % decodingFrequency) == 0)
                {
                    IDictionary<Variable, Value> outputs = new Dictionary<Variable, Value> { { decodingFunction, null } };
                    decodingFunction.Evaluate(new Dictionary<Variable, Value> {
                        { decodingFunction.Arguments[0], featureValue },
                        { decodingFunction.Arguments[1], labelValue } },
                        outputs,
                        device);
                    sequenceDataSource.PrintTranslation(featureValue, outputs[decodingFunction], rawLabels, decodingFunction, device);
                }
            }
        }

        private static Function BuildSeqToSeqModel(DeviceDescriptor device, bool useSparseInputs, int inputVocabDim, int labelVocabDim, int hiddenDim, int numLayers, int inputEmbeddingDim, int labelEmbeddingDim, Variable rawLabels, Function inputSequence, Function labelSequence, out Function decoderHistoryHook)
        {
            var labelSentenceStart = CNTKLib.SequenceFirst(rawLabels, "labelSequenceStart");

            var isFirstLabel = CNTKLib.SequenceIsFirst(labelSequence, "isFirstLabel");

            bool forceEmbedding = useSparseInputs;

            /* Embeddings */
            var inputEmbeddingWeights = new Parameter(new int[] { inputEmbeddingDim, NDShape.InferredDimension }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device, "inputEmbeddingWeights");
            var labelEmbeddingWeights = new Parameter(new int[] { labelEmbeddingDim, NDShape.InferredDimension }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device, "labelEmbeddingWeights");

            var inputEmbedding = CNTKLib.Alias((!forceEmbedding && (inputVocabDim <= inputEmbeddingDim)) ? inputSequence : CNTKLib.Times(inputEmbeddingWeights, inputSequence), "inputEmbedding");
            var labelEmbedding = CNTKLib.Alias((!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSequence : CNTKLib.Times(labelEmbeddingWeights, labelSequence), "labelEmbedding");
            var labelSentenceStartEmbedding = CNTKLib.Alias((!forceEmbedding && (labelVocabDim <= labelEmbeddingDim)) ? labelSentenceStart : CNTKLib.Times(labelEmbeddingWeights, labelSentenceStart), "labelSentenceStartEmbedding");
            var labelSentenceStartEmbeddedScattered = CNTKLib.SequenceScatter(labelSentenceStartEmbedding, isFirstLabel, "labelSentenceStartEmbeddedScattered");

            /* Encoder */
            var encoderOutputH = Stabilize<float>(inputEmbedding, device);
            Function encoderOutputC = null;
            Func<Variable, Function> futureValueRecurrenceHook = (x) => CNTKLib.FutureValue(x);

            for (int i = 0; i < numLayers; ++i)
            {
                Tuple<Function, Function> encoderOutputs = LSTMPComponentWithSelfStabilization<float>(encoderOutputH,
                    new int[] { hiddenDim }, new int[] { hiddenDim }, futureValueRecurrenceHook, futureValueRecurrenceHook, device);
                encoderOutputH = encoderOutputs.Item1;
                encoderOutputC = encoderOutputs.Item2;
            }

            var thoughtVectorH = CNTKLib.SequenceFirst(encoderOutputH, "thoughtVectorH");
            var thoughtVectorC = CNTKLib.SequenceFirst(encoderOutputC, "thoughtVectorC");

            bool addBeamSearchReorderingHook = true;
            if (addBeamSearchReorderingHook)
            {
                thoughtVectorH = CNTKLib.Reshape(thoughtVectorH, thoughtVectorH.Output.Shape.AppendShape(new int[] { 1 }), "thoughtVectorH");
                thoughtVectorC = CNTKLib.Reshape(thoughtVectorC, thoughtVectorC.Output.Shape.AppendShape(new int[] { 1 }), "thoughtVectorC");
                labelEmbedding = CNTKLib.Reshape(labelEmbedding, labelEmbedding.Output.Shape.AppendShape(new int[] { 1 }), "labelEmbedding");
                labelSentenceStartEmbeddedScattered = CNTKLib.Reshape(labelSentenceStartEmbeddedScattered,
                    labelSentenceStartEmbeddedScattered.Output.Shape.AppendShape(new int[] { 1 }), "labelSentenceStartEmbeddedScattered");
            }

            var thoughtVectorBroadcastH = CNTKLib.SequenceBroadcastAs(thoughtVectorH, labelEmbedding, "thoughtVectorBroadcastH");
            var thoughtVectorBroadcastC = CNTKLib.SequenceBroadcastAs(thoughtVectorC, labelEmbedding, "thoughtVectorBroadcastC");

            /* Decoder */
            var beamSearchReorderHook = new Constant(new int[] { 1, 1 }, 1.0f, device);
            var decoderHistoryFromGroundTruth = labelEmbedding;
            decoderHistoryHook = CNTKLib.Alias(decoderHistoryFromGroundTruth);
            var decoderInput = CNTKLib.ElementSelect(isFirstLabel, labelSentenceStartEmbeddedScattered, CNTKLib.PastValue(decoderHistoryHook));

            var decoderOutputH = Stabilize<float>(decoderInput, device);
            Func<Variable, Function> pastValueRecurrenceHookWithBeamSearchReordering = (operand) =>
                CNTKLib.PastValue(addBeamSearchReorderingHook ? CNTKLib.Times(operand, beamSearchReorderHook) : operand.ToFunction());

            for (int i = 0; i < numLayers; ++i)
            {
                Func<Variable, Function> recurrenceHookH, recurrenceHookC;
                var isFirst = CNTKLib.SequenceIsFirst(labelEmbedding);
                if (i > 0)
                {
                    recurrenceHookH = pastValueRecurrenceHookWithBeamSearchReordering;
                    recurrenceHookC = pastValueRecurrenceHookWithBeamSearchReordering;
                }
                else
                {
                    recurrenceHookH = (operand) =>
                        CNTKLib.ElementSelect(isFirst, thoughtVectorBroadcastH,
                        CNTKLib.PastValue(addBeamSearchReorderingHook ? CNTKLib.Times(operand, beamSearchReorderHook) : operand.ToFunction()));

                    recurrenceHookC = (operand) =>
                        CNTKLib.ElementSelect(isFirst, thoughtVectorBroadcastC,
                        CNTKLib.PastValue(addBeamSearchReorderingHook ? CNTKLib.Times(operand, beamSearchReorderHook) : operand.ToFunction()));
                }

                NDShape outDims = new int[] { hiddenDim };
                NDShape cellDims = new int[] { hiddenDim };
                if (addBeamSearchReorderingHook)
                {
                    outDims = outDims.AppendShape(new int[] { 1 });
                    cellDims = cellDims.AppendShape(new int[] { 1 });
                }
                var decoderOutputs = LSTMPComponentWithSelfStabilization<float>(decoderOutputH, outDims, cellDims, recurrenceHookH, recurrenceHookC, device);
                decoderOutputH = decoderOutputs.Item1;
                encoderOutputC = decoderOutputs.Item2;
            }

            var decoderOutput = decoderOutputH;
            var decoderDim = hiddenDim;

            /* Softmax output layer */
            var outputLayerProjWeights = new Parameter(new int[] { labelVocabDim, decoderDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
            var biasWeights = new Parameter(new int[] { labelVocabDim }, 0.0f, device, "");

            return CNTKLib.Plus(CNTKLib.Times(outputLayerProjWeights, Stabilize<float>(decoderOutput, device)), biasWeights, "classifierOutput");
        }
    }
}

