﻿//
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
using System.Threading;
using System.Threading.Tasks;
using CNTK;
using CNTKExtension;
using CNTKImageProcessing;

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
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
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

                // Image preprocessing to match input requirements of the model.
                // This program uses images from the CIFAR-10 dataset for evaluation.
                // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
                string sampleImage = "00000.png";
                ThrowIfFileNotExist(sampleImage, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", sampleImage));
                Bitmap bmp = new Bitmap(Bitmap.FromFile(sampleImage));
                var resized = bmp.Resize(imageWidth, imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();

                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(inputShape, resizedCHW, device);
                inputDataMap.Add(inputVar, inputVal);

                // The model has only one output.
                // You can also use the following way to get output variable by name:
                // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                Console.WriteLine("Evaluation result for image " + sampleImage);
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
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                Function modelFunc = Function.Load(modelFilePath, device);

                // Get input variable. The model has only one single input.
                // The same way described above for output variable can be used here to get input variable by name.
                Variable inputVar = modelFunc.Arguments.Single();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];

                Bitmap bmp, resized;
                List<float> resizedCHW;
                var seqData = new List<float>();
                for (int sampleIndex = 0; sampleIndex < imageList.Count; sampleIndex++)
                {
                    bmp = new Bitmap(Bitmap.FromFile(imageList[sampleIndex]));
                    resized = bmp.Resize(imageWidth, imageHeight, true);
                    resizedCHW = resized.ParallelExtractCHW();
                    // Aadd this sample to the data buffer.
                    seqData.AddRange(resizedCHW);
                }

                // Create Value for the batch data.
                var inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);
                // Create input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                inputDataMap.Add(inputVar, inputVal);

                // The model has only one output.
                // You can also use the following way to get output variable by name:
                // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evaluate the model against the batch input
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Retrieve the evaluation result.
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // Output result
                Console.Write("Evaluation result for batch of images: ");
                for (int index = 0; index < imageList.Count; index++)
                {
                    Console.Write(imageList[index]);
                    if (index < imageList.Count - 1)
                        Console.Write(", ");
                    else
                        Console.WriteLine();
                }

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
        public static async Task EvaluateMultipleImagesInParallelAsync(DeviceDescriptor device)
        {
            Console.WriteLine("\n===== Evaluate multiple images in parallel =====");

            string modelFilePath = "resnet20.dnn";
            ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));

            // This program uses images from the CIFAR-10 dataset for evaluation.
            // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
            var imageFiles = new string[] { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png" };
            var imageList = new BlockingCollection<string>();
            foreach (var file in imageFiles)
            {
                ThrowIfFileNotExist(file, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", file));
                // For simplicity, we add all images to the BlockingCollection in advance. It is also possible to add new images dynamically.
                imageList.Add(file);
            }

            // Load and clone the model.
            // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
            // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
            var modelFunc = Function.Load(modelFilePath, device);

            // It is not thread-safe to perform concurrent evaluation requests using the same model function.
            // Use clone() to create copies of model function for parallel evaluation.
            // ParameterCloningMethod.Share specifies that model parameters are shared between cloned model functions, while
            // each model function instance has its own private state for evaluation.
            int numOfModelInstances = 3;
            List<Function> modelPool = new List<Function>();
            modelPool.Add(modelFunc);

            for (int i = 1; i < numOfModelInstances; i++)
            {
                modelPool.Add(modelFunc.Clone(ParameterCloningMethod.Share));
            }

            // Start to evaluate samples in parallel.
            Console.WriteLine(string.Format("Evaluate {0} images in parallel using {1} model instances.", imageList.Count, numOfModelInstances));
            var taskList = new List<Task>();
            var results = new ConcurrentDictionary<string, IList<IList<float>>>();
            foreach (var evalFunc in modelPool)
            {
                taskList.Add(Task.Factory.StartNew(() =>
                {
                    // Get input and output variables
                    Variable inputVar = evalFunc.Arguments.Single();
                    NDShape inputShape = inputVar.Shape;
                    int imageWidth = inputShape[0];
                    int imageHeight = inputShape[1];
                    Variable outputVar = evalFunc.Output;

                    string image;
                    // The task exits when no image is available for evaluation.
                    while (imageList.TryTake(out image) == true)
                    {
                        Console.WriteLine(string.Format("Evaluating image {0} using thread {1}.", image, Thread.CurrentThread.ManagedThreadId));

                        Bitmap bmp = new Bitmap(Bitmap.FromFile(image));
                        var resized = bmp.Resize(imageWidth, imageHeight, true);
                        List<float> resizedCHW = resized.ParallelExtractCHW();

                        // Create input data map.
                        var inputDataMap = new Dictionary<Variable, Value>();
                        var inputVal = Value.CreateBatch(inputShape, resizedCHW, device);
                        inputDataMap.Add(inputVar, inputVal);

                        // Create output data map.
                        var outputDataMap = new Dictionary<Variable, Value>();
                        outputDataMap.Add(outputVar, null);

                        // Start evaluation on the device
                        evalFunc.Evaluate(inputDataMap, outputDataMap, device);

                        // Get evaluate result as dense output
                        var outputVal = outputDataMap[outputVar];
                        var outputData = outputVal.GetDenseData<float>(outputVar);

                        // Add result to the buffer for output at a later time.
                        if (results.TryAdd(image, outputData) == false)
                           throw new ArgumentException(string.Format("The image {0} has already been evaluated.", image));
                   }
               }));
            }

            // Await until all images have been evaluated.
            await Task.WhenAll(taskList);

            var sampleSize = modelFunc.Output.Shape.TotalSize;
            foreach (var file in imageFiles)
            {
                if (!results.ContainsKey(file))
                    throw new KeyNotFoundException(string.Format("Error: the image {0} has not been evaluated.", file));
                var evalResult = results[file];
                Console.WriteLine("Evaluation result for image " + file);
                PrintOutput(sampleSize, evalResult);
            }
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
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                string modelFilePath = "resnet20.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                var modelBuffer = File.ReadAllBytes(modelFilePath);

                // Load model from memroy buffer
                Function modelFunc = Function.Load(modelBuffer, device);
                
                // Get shape data for the input variable
                Variable inputVar = modelFunc.Arguments.Single();
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];

                // Image preprocessing to match input requirements of the model.
                // This program uses images from the CIFAR-10 dataset for evaluation.
                // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
                string sampleImage = "00000.png";
                ThrowIfFileNotExist(sampleImage, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", sampleImage));
                Bitmap bmp = new Bitmap(Bitmap.FromFile(sampleImage));
                var resized = bmp.Resize(imageWidth, imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();

                // Create input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
                inputDataMap.Add(inputVar, inputVal);

                // Get output variable.
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device.
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output.
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
        /// - how to evaluate model using asynchronous task. This is useful when offloading is needed to achieve better responsiveness.
        /// The asynchronous evaluation is implemented as an extension method in CNTKExtensions.cs, which provides an asynchrounous facade for the synchronous Evaluation().
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation.</param>
        public static async Task EvaluationSingleImageAsync(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate image asynchronously =====");

                // Load the model.
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/Models/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                string modelFilePath = "resnet20.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                Function modelFunc = Function.Load(modelFilePath, device);

                // Get input variable.
                Variable inputVar = modelFunc.Arguments.Single();

                // Get shape data for the input variable.
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];

                // Image preprocessing to match input requirements of the model.
                // This program uses images from the CIFAR-10 dataset for evaluation.
                // Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.
                string sampleImage = "00000.png";
                ThrowIfFileNotExist(sampleImage, string.Format("Error: The sample image '{0}' does not exist. Please see README.md in <CNTK>/Examples/Image/DataSets/CIFAR-10 about how to download the CIFAR-10 dataset.", sampleImage));
                Bitmap bmp = new Bitmap(Bitmap.FromFile(sampleImage));
                var resized = bmp.Resize(imageWidth, imageHeight, true);
                List<float> resizedCHW = resized.ParallelExtractCHW();

                // Create input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(inputShape, resizedCHW, device);
                inputDataMap.Add(inputVar, inputVal);

                // Get output variable.
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Start evaluation, await on the result.
                await modelFunc.EvaluateAsync(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output.
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
                // SeqStartFlag is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
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

                // Prepare output.
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evaluate the model.
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get output result.
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // output the result.
                var outputSampleSize = outputVar.Shape.TotalSize;
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

                // Get input variable.
                var inputVar = modelFunc.Arguments.Single();
                int vocabSize = inputVar.Shape.TotalSize;

                // Prepare input data.
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
                var inputValue = Value.CreateBatchOfSequences<float>(vocabSize, inputBatch, seqStartFlagBatch, device);

                // Build input data map.
                var inputDataMap = new Dictionary<Variable, Value>();
                inputDataMap.Add(inputVar, inputValue);

                // Prepare output.
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evaluate the model
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluation result.
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // output the result
                var outputSampleSize = outputVar.Shape.TotalSize;
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
                    var nonZeroValueIndex = vocabToIndex[inputWords[count]];
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

                // Create output data map. Using null as Value to indicate using system allocated memory.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Evaluate the model.
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get result
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                // Output the result
                var outputSampleSize = outputVar.Shape.TotalSize;
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
        /// - how to load a pretrained model and evaluate an intermediate layer of its network
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation</param>
        public static void EvaluateIntermediateLayer(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate intermediate layer =====\n");

                // Load the model.
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                string modelFilePath = "resnet20.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                Function rootFunc = Function.Load(modelFilePath, device);

                Function interLayerPrimitiveFunc = rootFunc.FindByName("final_avg_pooling");

                // The Function returned by FindByName is a primitive function.
                // For evaluation, it is required to create a composite function from the primitive function.
                Function modelFunc = Function.AsComposite(interLayerPrimitiveFunc);

                Variable outputVar = modelFunc.Output;
                Variable inputVar = modelFunc.Arguments.Single();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];
                int imageChannels = inputShape[2];
                int imageSize = inputShape.TotalSize;

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

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output
                var outputVal = outputDataMap[outputVar];
                var outputData = outputVal.GetDenseData<float>(outputVar);

                Console.WriteLine("Evaluation result of intermediate layer final_avg_pooling");
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
        /// - how to load a pretrained model and evaluate several nodes by combining their outputs
        /// </summary>
        /// <param name="device">Specify on which device to run the evaluation</param>
        public static void EvaluateCombinedOutputs(DeviceDescriptor device)
        {
            try
            {
                Console.WriteLine("\n===== Evaluate combined outputs =====\n");

                // Load the model.
                // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
                // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
                string modelFilePath = "resnet20.dnn";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));
                Function modelFunc = Function.Load(modelFilePath, device);

                // Get node of interest
                Function interLayerPrimitiveFunc = modelFunc.FindByName("final_avg_pooling");
                Variable poolingOutput = interLayerPrimitiveFunc.Output;

                // Create a function which combine outputs from the node "final_avg_polling" and the final layer of the model.
                Function evalFunc = Function.Combine(new[] { modelFunc.Output, poolingOutput });
                Variable inputVar = evalFunc.Arguments.Single();

                // Get shape data for the input variable
                NDShape inputShape = inputVar.Shape;
                int imageWidth = inputShape[0];
                int imageHeight = inputShape[1];
                int imageChannels = inputShape[2];
                int imageSize = inputShape.TotalSize;

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

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var modelOutput = evalFunc.Outputs[0];
                var interLayerOutput = evalFunc.Outputs[1];

                outputDataMap.Add(modelOutput, null);
                outputDataMap.Add(interLayerOutput, null);

                // Start evaluation on the device
                evalFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output
                foreach (var outputVariableValuePair in outputDataMap)
                {
                    var variable = outputVariableValuePair.Key;
                    var value = outputVariableValuePair.Value;
                    var outputData = value.GetDenseData<float>(variable);

                    string variableName = "last layer of the model";
                    if (variable.Name == interLayerPrimitiveFunc.Name) {
                        variableName = "intermediate layer " + variable.Name;
                    }
                    
                    Console.WriteLine("Evaluation result of {0}", variableName);
                    PrintOutput(variable.Shape.TotalSize, outputData);
                }
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
