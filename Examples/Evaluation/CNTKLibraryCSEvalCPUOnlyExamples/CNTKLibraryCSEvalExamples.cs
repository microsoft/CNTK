//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCSEvalExamples.cs -- Examples for using CNTK Library C# Eval API.
//

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    public class CNTKLibraryManagedExamples
    {
        /// <summary>
        /// The example shows
        /// - how to load model.
        /// - how to prepare input data for a single sample.
        /// - how to prepare input and output data map.
        /// - how to evaluate a model.
        /// - how to retrieve evaluation result and retrieve output data in dense format.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation.</param>
        public static void EvaluationSingleImage(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate single image =====");

                // Load the model.
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                string modelFilePath = "resnet20.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                Function modelFunc = Function.Load(modelFilePath, device);

                // Get input variable. The model has only one single input.
                // The same way described above for output variable can be used here to get input variable by name.
                Variable inputVar = modelFunc.Arguments.Single();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];
                int imageChannels = inputShape[2];
                int imageSize = inputShape.TotalSize;

                // The model has only one output.
                // If the model have more than one output, use the following way to get output variable by name.
                // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
                Variable outputVar = modelFunc.Output;

                var inputDataMap = new Dictionary<Variable, Value>();
                var outputDataMap = new Dictionary<Variable, Value>();

                // Image preprocessing to match input requirements of the model.
                // This program uses images from the CIFAR-10 dataset for evaluation.
                // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
                string sampleImage = "00000.png";
                ThrowIfFileNotExist(sampleImage, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", sampleImage));
                Bitmap bmp = new Bitmap(Bitmap.FromFile(sampleImage));
                var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();

                // Create input data map
                var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
                inputDataMap.Add(inputVar, inputVal);

                // Create ouput data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                PrintOutput(outputVar.Shape.TotalSize, outputData);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        /// <summary>
        /// The example shows
        /// - how to load model.
        /// - how to prepare input data for a batch of samples.
        /// - how to prepare input and output data map.
        /// - how to evaluate a model.
        /// - how to retrieve evaluation result and retrieve output data in dense format.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation.</param>
        public static void EvaluationBatchOfImages(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate batch of images =====");

                string modelFilePath = "resnet20.dnn";
                // This program uses images from the CIFAR-10 dataset for evaluation.
                // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
                var imageList = new List<string>() { "00000.png", "00001.png", "00002.png" };
                foreach (var image in imageList)
                {
                    ThrowIfFileNotExist(image, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", image));
                }
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));

                // Load the model.
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                Function modelFunc = Function.Load(modelFilePath, device);

                // Get input variable. The model has only one single input.
                // The same way described above for output variable can be used here to get input variable by name.
                Variable inputVar = modelFunc.Arguments.Single();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];
                int imageChannels = inputShape[2];
                int imageSize = inputShape.TotalSize;

                // The model has only one output.
                // If the model have more than one output, use the following way to get output variable by name.
                // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
                Variable outputVar = modelFunc.Output;

                var inputDataMap = new Dictionary<Variable, Value>();
                var outputDataMap = new Dictionary<Variable, Value>();

                Bitmap bmp, resized;
                List<float> resizedCHW;
                var seqData = new List<float>();
                for (int sampleIndex = 0; sampleIndex < imageList.Count; sampleIndex++)
                {
                    bmp = new Bitmap(Bitmap.FromFile(imageList[sampleIndex]));
                    resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    resizedCHW = resized.ParallelExtractCHW();
                    // Aadd this sample to the data buffer.
                    seqData.AddRange(resizedCHW);
                }

                // Create Value for the batch data.
                var inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);
                // Create input data map.
                inputDataMap.Add(inputVar, inputVal);

                // Create ouput data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                outputDataMap.Add(outputVar, null);

                // Evaluate the model against the batch input
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Retrieve the evaluation result.
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // Output result
                PrintOutput(outputVar.Shape.TotalSize, outputData);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        /// <summary>
        /// The example shows
        /// - how to evaluate multiple sample requests in parallel.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation.</param>
        public static void EvaluateMultipleImagesInParallel(DeviceDescriptor device)
        {
            Console.WriteLine("\n===== Evaluate multiple images in parallel =====");

            string modelFilePath = "resnet20.dnn";

            // This program uses images from the CIFAR-10 dataset for evaluation.
            // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
            var imageList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png" };
            foreach (var image in imageList)
            {
                ThrowIfFileNotExist(image, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", image));
            }
            ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));

            int maximalNumOfParallelRequests = 3;
            BlockingCollection<Function> Models = new BlockingCollection<Function>();

            // Load and clone the model.
            // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            var rootFunc = Function.Load(modelFilePath, device);
            Models.Add(rootFunc);

            // It is not thread-safe to perform concurrent evaluation requests using the same model function.
            // Use clone() to create copies of model function for parallel evaluation.
            // ParameterCloningMethod.Share specifies that model parameters are shared between cloned model functions, while
            // each model function instance has its own private state for evaluation.
            for (int i = 1; i < maximalNumOfParallelRequests; i++)
            {
                Models.Add(rootFunc.Clone(ParameterCloningMethod.Share));
            }

            // Get shape data for the input variable
            var input = rootFunc.Arguments.Single();
            NDShape inputShape = input.Shape;
            int imageWidth = inputShape[0];
            int imageHeight = inputShape[1];
            int imageChannels = inputShape[2];
            int imageSize = inputShape.TotalSize;
            Object lockObj = new object();

            // Start to evaluate samples in parallel.
            // If there are more evaluation requests than the number of available model function instances, some evaluation
            // requests will have to wait for a free model function instance.
            Console.WriteLine(string.Format("Evaluate {0} images in parallel using {1} model instances.", imageList.Count, maximalNumOfParallelRequests));
            Parallel.ForEach(imageList, new ParallelOptions() { MaxDegreeOfParallelism = imageList.Count }, (image) =>
            {
                var evaluatorFunc = Models.Take();
                try
                {
                    Variable outputVar = evaluatorFunc.Output;
                    Variable inputVar = evaluatorFunc.Arguments.Single();
                    var inputDataMap = new Dictionary<Variable, Value>();
                    var outputDataMap = new Dictionary<Variable, Value>();

                    Bitmap bmp = new Bitmap(Bitmap.FromFile(image));
                    var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();

                    // Create input data map
                    var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
                    inputDataMap.Add(inputVar, inputVal);

                    // Create ouput data map. Using null as Value to indicate using system allocated memory.
                    // Alternatively, create a Value object and add it to the data map.
                    outputDataMap.Add(outputVar, null);

                    // Start evaluation on the device
                    evaluatorFunc.Evaluate(inputDataMap, outputDataMap, device);

                    // Get evaluate result as dense output
                    var outputVal = outputDataMap[outputVar];
                    var outputData = outputVal.GetDenseData<float>(outputVar);

                    // Serialize output
                    lock (lockObj)
                    {
                        Console.WriteLine(string.Format("Evaluation result for {0}:", image));
                        PrintOutput(outputVar.Shape.TotalSize, outputData);
                    }
                }
                finally
                {
                    Models.Add(evaluatorFunc);
                }
            });
        }

        /// <summary>
        /// The example shows
        /// - how to load model from a memory buffer.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation.</param>
        public static void LoadModelFromMemory(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Load model from memory buffer =====");

                // For demo purpose, we first read the the model into memory
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                string modelFilePath = "resnet20.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                var modelBuffer = File.ReadAllBytes(modelFilePath);

                // Load model from memroy buffer
                Function modelFunc = Function.Load(modelBuffer, device);

                // Get input variable. The model has only one single input.
                // The same way described above for output variable can be used here to get input variable by name.
                Variable inputVar = modelFunc.Arguments.Single();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];
                int imageChannels = inputShape[2];
                int imageSize = inputShape.TotalSize;

                // The model has only one output.
                // If the model have more than one output, use the following way to get output variable by name.
                // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
                Variable outputVar = modelFunc.Output;

                var inputDataMap = new Dictionary<Variable, Value>();
                var outputDataMap = new Dictionary<Variable, Value>();

                // Image preprocessing to match input requirements of the model.
                // This program uses images from the CIFAR-10 dataset for evaluation.
                // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
                string sampleImage = "00000.png";
                ThrowIfFileNotExist(sampleImage, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", sampleImage));
                Bitmap bmp = new Bitmap(Bitmap.FromFile(sampleImage));
                var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();

                // Create input data map
                var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
                inputDataMap.Add(inputVar, inputVal);

                // Create ouput data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                PrintOutput(outputVar.Shape.TotalSize, outputData);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        /// <summary>
        /// Print out the evalaution results.
        /// </summary>
        /// <typeparam name="T">The data value type</typeparam>
        /// <param name="sampleSize">The size of each sample.</param>
        /// <param name="outputBuffer">The evaluation result data.</param>
        internal static void PrintOutput<T>(int sampleSize, IList<IList<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            int outputSampleSize = sampleSize;
            foreach (var seq in outputBuffer)
            {
                if (seq.Count % outputSampleSize != 0)
                {
                    throw new ApplicationException("The number of elements in the sequence is not a multiple of sample size");
                }

                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                int i = 0;
                int sampleNo = 0;
                foreach (var element in seq)
                {
                    if (i++ % outputSampleSize == 0)
                    {
                        Console.Write(String.Format("    sample {0}: ", sampleNo));
                    }
                    Console.Write(element);
                    if (i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                    {
                        Console.Write(",");
                    }
                }
            }
        }

        /// <summary>
        /// The example shows
        /// - how to load model.
        /// - how to prepare input data as sequence using one-hot vector.
        /// - how to prepare input and output data map.
        /// - how to evaluate a model.
        /// - how to retrieve evaluation result.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation</param>
        public static void EvaluationSingleSequenceUsingOneHot(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate single sequence using one-hot vector =====");

                // The model atis.dnn is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
                // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
                string modelFilePath = "atis.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS to create the model.", modelFilePath));
                Function modelFunc = Function.Load(modelFilePath, device);

                // Read word and slot index files.
                string vocabFile = "query.wl";
                string labelFile = "slots.wl";
                ThrowIfFileNotExist(vocabFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/ to the output directory.", vocabFile));
                ThrowIfFileNotExist(labelFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/ to the output directory.", labelFile));
                var vocabToIndex = buildVocabIndex(vocabFile);
                var indexToSlots = buildSlotIndex(labelFile);

                // Get input variable
                var inputVar = modelFunc.Arguments.Single();
                int vocabSize = inputVar.Shape.TotalSize;

                var inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
                var seqData = new List<int>();
                // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
                var seqStartFlag = true;
                // Get the index of each word in the sentence.
                string[] inputWords = inputSentence.Split(' ');
                foreach (var str in inputWords)
                {
                    // Get the index of the word
                    var index = vocabToIndex[str];
                    // Add the sample to the sequence
                    seqData.Add(index);
                }

                // Create input value using OneHot vector data.
                var inputValue = Value.CreateSequence<float>(vocabSize, seqData, seqStartFlag, device);

                // Build input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                inputDataMap.Add(inputVar, inputValue);

                // Prepare output
                Variable outputVar = modelFunc.Output;

                // Create ouput data map. Using null as Value to indicate using system allocated memory.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evalaute the model.
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get output result
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // output the result
                var outputSampleSize = (int)outputVar.Shape.TotalSize;
                if (outputData.Count != 1)
                {
                    throw new ApplicationException("Only one sequence of slots is expected as output.");
                }
                var slotSeq = outputData[0];
                if (slotSeq.Count % outputSampleSize != 0)
                {
                    throw new ApplicationException("The number of elements in the slot sequence is not a multiple of sample size");
                }

                var numOfSlotsInOutput = slotSeq.Count / outputSampleSize;
                if (inputWords.Count() != numOfSlotsInOutput)
                {
                    throw new ApplicationException("The number of input words and the number of output slots do not match");
                }
                for (int i = 0; i < numOfSlotsInOutput; i++)
                {
                    var max = slotSeq[i * outputSampleSize];
                    var maxIndex = 0;
                    for (int j = 1; j < outputSampleSize; j++)
                    {
                        if (slotSeq[i * outputSampleSize + j] > max)
                        {
                            max = slotSeq[i * outputSampleSize + j];
                            maxIndex = j;
                        }
                    }
                    Console.WriteLine(String.Format("     {0, 10} ---- {1}", inputWords[i], indexToSlots[maxIndex]));
                }
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        /// <summary>
        /// The example shows
        /// - how to load model.
        /// - how to prepare input data as batch of sequences with variable length.
        ///   how to prepare data using one-hot vector format.
        /// - how to prepare input and output data map.
        /// - how to evaluate a model.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation.</param>
        public static void EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate batch of sequences with variable length using one-hot vector =====");

                // The model atis.dnn is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
                // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
                string modelFilePath = "atis.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS to create the model.", modelFilePath));
                Function modelFunc = Function.Load(modelFilePath, device);

                // Read word and slot index files.
                string vocabFile = "query.wl";
                string labelFile = "slots.wl";
                ThrowIfFileNotExist(vocabFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/ to the output directory.", vocabFile));
                ThrowIfFileNotExist(labelFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/ to the output directory.", labelFile));
                var vocabToIndex = buildVocabIndex(vocabFile);
                var indexToSlots = buildSlotIndex(labelFile);

                // Get input variable
                var inputVar = modelFunc.Arguments.Single();
                int vocabSize = inputVar.Shape.TotalSize;

                // Prepare the input data.
                // Each sample is represented by an index to the onehot vector, so the index of the non-zero value of each sample is saved in the inner list.
                // The outer list represents sequences contained in the batch.
                var inputBatch = new List<List<int>>();
                // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
                var seqStartFlagBatch = new List<bool>();

                var inputSentences = new List<string>() {
                    "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                    "BOS flights from new york to seattle EOS"
                };

                var inputWords = new List<string[]>(2);
                int numOfSequences = inputSentences.Count;
                for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
                {
                    // The input for one sequence
                    // Get the index of each word in the sentence.
                    var substring = inputSentences[seqIndex].Split(' ');
                    inputWords.Add(substring);
                    var seqData = new List<int>();
                    foreach (var str in substring)
                    {
                        var index = vocabToIndex[str];
                        seqData.Add(index);
                    }
                    inputBatch.Add(seqData);
                    seqStartFlagBatch.Add(true);
                }

                // Create the Value representing the batch data.
                var inputValue = Value.CreateBatchOfSequences<float>(vocabSize, inputBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);

                // Build input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                inputDataMap.Add(inputVar, inputValue);

                // Prepare output
                Variable outputVar = modelFunc.Output;
                // Create ouput data map. Using null as Value to indicate using system allocated memory.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evalaute the model
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluation result.
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // output the result
                var outputSampleSize = (int)outputVar.Shape.TotalSize;
                if (outputData.Count != inputBatch.Count)
                {
                    throw new ApplicationException("The number of sequence in output does not match that in input.");
                }
                Console.WriteLine("The number of sequences in the batch: " + outputData.Count);

                for (int seqno = 0; seqno < outputData.Count; seqno++)
                {
                    var slotSeq = outputData[seqno];
                    Console.WriteLine("Sequence {0}: ", seqno);

                    if (slotSeq.Count % outputSampleSize != 0)
                    {
                        throw new ApplicationException("The number of elements in the slot sequence is not a multiple of sample size");
                    }

                    var numOfSlotsInOutput = slotSeq.Count / outputSampleSize;
                    if (inputWords[seqno].Count() != numOfSlotsInOutput)
                    {
                        throw new ApplicationException("The number of input words and the number of output slots do not match.");
                    }
                    for (int i = 0; i < numOfSlotsInOutput; i++)
                    {
                        var max = slotSeq[i * outputSampleSize];
                        var maxIndex = 0;
                        for (int j = 1; j < outputSampleSize; j++)
                        {
                            if (slotSeq[i * outputSampleSize + j] > max)
                            {
                                max = slotSeq[i * outputSampleSize + j];
                                maxIndex = j;
                            }
                        }
                        Console.WriteLine(String.Format("     {0, 10} ---- {1}", inputWords[seqno][i], indexToSlots[maxIndex]));
                    }
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        /// <summary>
        /// The example shows
        /// - how to prepare input data as sequence using sparse input.
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation</param>
        public static void EvaluationSingleSequenceUsingSparse(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate single sequence using sparse input =====");

                // The model atis.dnn is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
                // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
                string modelFilePath = "atis.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS to create the model.", modelFilePath));
                Function modelFunc = Function.Load(modelFilePath, device);

                // Read word and slot index files.
                string vocabFile = "query.wl";
                string labelFile = "slots.wl";
                ThrowIfFileNotExist(vocabFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/ to the output directory.", vocabFile));
                ThrowIfFileNotExist(labelFile, string.Format("Error: The file '{0}' does not exist. Please copy it from <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/ to the output directory.", labelFile));
                var vocabToIndex = buildVocabIndex(vocabFile);
                var indexToSlots = buildSlotIndex(labelFile);

                // Get input variable
                var inputVar = modelFunc.Arguments.Single();
                int vocabSize = inputVar.Shape.TotalSize;

                var inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";

                // Get the index of each word in the sentence.
                string[] inputWords = inputSentence.Split(' ');
                var seqLen = inputWords.Length;
                // For this example, only 1 non-zero value for each sample.
                var numNonZeroValues = seqLen * 1;
                var colStarts = new int[seqLen + 1];
                var rowIndices = new int[numNonZeroValues];
                var nonZeroValues = new float[numNonZeroValues];

                int count = 0;
                for (; count < seqLen; count++)
                {
                    // Get the index of the word
                    var nonZeroValueIndex = (int)vocabToIndex[inputWords[count]];
                    // Add the sample to the sequence
                    nonZeroValues[count] = (float)1.0;
                    rowIndices[count] = nonZeroValueIndex;
                    colStarts[count] = count;
                }
                colStarts[count] = numNonZeroValues;

                // Create input value using OneHot vector data.
                var inputValue = Value.CreateSequence<float>(vocabSize, seqLen, colStarts, rowIndices, nonZeroValues, device);

                // Build input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                inputDataMap.Add(inputVar, inputValue);

                // Prepare output
                Variable outputVar = modelFunc.Output;

                // Create ouput data map. Using null as Value to indicate using system allocated memory.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evalaute the model.
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get result
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // Output the result
                var outputSampleSize = (int)outputVar.Shape.TotalSize;
                if (outputData.Count != 1)
                {
                    throw new ApplicationException("Only one sequence of slots is expected as output.");
                }
                var slotSeq = outputData[0];
                if (slotSeq.Count % outputSampleSize != 0)
                {
                    throw new ApplicationException("The number of elements in the slot sequence is not a multiple of sample size");
                }

                var numOfSlotsInOutput = slotSeq.Count / outputSampleSize;
                if (inputWords.Count() != numOfSlotsInOutput)
                {
                    throw new ApplicationException("The number of input words and the number of output slots do not match");
                }
                for (int i = 0; i < numOfSlotsInOutput; i++)
                {
                    var max = slotSeq[i * outputSampleSize];
                    var maxIndex = 0;
                    for (int j = 1; j < outputSampleSize; j++)
                    {
                        if (slotSeq[i * outputSampleSize + j] > max)
                        {
                            max = slotSeq[i * outputSampleSize + j];
                            maxIndex = j;
                        }
                    }
                    Console.WriteLine(String.Format("     {0, 10} ---- {1}", inputWords[i], indexToSlots[maxIndex]));
                }
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        /// <summary>
        /// Checks whether the file exists. If not, write the error message on the console and throw FileNotFoundException.
        /// </summary>
        /// <param name="filePath">The file to check.</param>
        /// <param name="errorMsg">The message to write on console if the file does not exist.</param>
        internal static void ThrowIfFileNotExist(string filePath, string errorMsg)
        {
            if (!File.Exists(filePath))
            {
                if (!string.IsNullOrEmpty(errorMsg))
                {
                    Console.WriteLine(errorMsg);
                }
                throw new FileNotFoundException(string.Format("File '{0}' not found.", filePath));
            }
        }

        private static Dictionary<string, int> buildVocabIndex(string filePath)
        {
            var vocab = new Dictionary<string, int>();

            string[] lines = File.ReadAllLines(filePath);
            for (int idx = 0; idx < lines.Count(); idx++)
                vocab.Add(lines[idx], idx);

            return vocab;
        }

        private static string[] buildSlotIndex(string filePath)
        {
            return File.ReadAllLines(filePath);
        }
    }
}
