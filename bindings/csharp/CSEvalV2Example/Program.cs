//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// program.cs -- Example for using C# Eval V2 API.
//

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CSEvalV2Example
{
    public class Program
    {
        // 
        // The example shows 
        // - how to load model.
        // - how to prepare input data for a signle sample, a batch of samples in dense format.
        // - how to prepare input and output data map
        // - how to evaluate a model
        // - how to retrieve evaluation result and retrieve output data in dense format.
        //
        static void EvaluationWithDenseData(DeviceDescriptor device)
        {
            const string outputName = "Plus2060_output";

            // Load the model.
            Function modelFunc = Function.LoadModel("z.model");

            // Get output variable based on name
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
 
            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.Single();

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            var outputDataMap = new Dictionary<Variable, Value>();

            // Use case 1: Evaluate with single image
            Console.WriteLine("Evaluate single image");

            // Image preprocessing to match input requirements of the model.
            Bitmap bmp = new Bitmap(Bitmap.FromFile("00000.png"));
            var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            var inputDataMap = new Dictionary<Variable, Value>();
            var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            var outputData = new List<List<float>>();
            Value outputVal = outputDataMap[outputVar];
            outputVal.CopyTo(outputVar, outputData);

            // Use case 2: Evaluate with batch of images
            Console.WriteLine("Evaluate batch of images");

            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png" };
            var seqData = new List<float>();
            for (int sampleIndex = 0; sampleIndex < fileList.Count; sampleIndex++)
            {
                bmp = new Bitmap(Bitmap.FromFile(fileList[sampleIndex++]));
                resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                resizedCHW = resized.ParallelExtractCHW();
                // Aadd this sample to the data buffer.
                seqData.AddRange(resizedCHW);
            }

            // Create Value for the batch data.
            inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);

            // Create input and output data map.
            inputDataMap[inputVar] = inputVal;
            outputDataMap[outputVar] = null;

            // Evaluate the model against the batch input
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Retrieve the evaluation result.
            outputData = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyTo(outputVar, outputData);
            
            // Output result
            Console.WriteLine("The number of sequences in the batch: " + outputData.Count);
            int seqNo = 0;
            uint outputSampleSize = outputVar.Shape.TotalSize;
            foreach(var seq in outputData)
            {
                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count/outputSampleSize));
                uint i = 0;
                uint sampleNo = 0;
                foreach (var element in seq)
                {
                    Console.Write(String.Format("    sample {0}: " + sampleNo));
                    Console.Write(element);
                    if (++i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                        Console.WriteLine(",");
                }
            }

        }

        // 
        // The example shows 
        // - how to use OneHot vector as input and output for evaluation
        //   The input data contains multiple sequences and each sequence contains multiple samples.
        //   There is only one non-zero value in each sample, so the sample can be represented by the index of this non-zero value
        // - use variable name, instead of Variable, for as parameters for evaluate.
        //
        static void EvaluationWithOneHot(DeviceDescriptor device)
        {
            // Todo: fill both index values
            var vocabToIndex = new Dictionary<string, uint>();
            var indexToVocab = new Dictionary<uint, string>();

            Function myFunc = Function.LoadModel("atis.model");

            // Get input variable 
            const string inputName = "features";
            var inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputName)).Single();

            uint vocabSize = inputVar.Shape.TotalSize;

            // Use case 1: Evalaute a single sequence using OneHot vector as input.
            var inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
            // Build input data for one sequence 
            var seqData = new List<uint>();
            // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
            var seqStartFlag = true;
            // Get the index of each word in the sentence.
            string[] substring = inputSentence.Split(' ');
            foreach (var str in substring)
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
            const string outputName = "out.z_output";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            var outputDataMap = new Dictionary<Variable, Value>();
            outputDataMap.Add(outputVar, null);

            // Evalaute the model.
            myFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get output result
            var outputData = new List<List<uint>>();
            Value outputVal = outputDataMap[outputVar];
            outputVal.CopyTo(outputVar, outputData);

            // Use case 2: evaluate batch of sequences using OneHot vector as input.

            // Prepare the input data. 
            // Each sample is represented by an index to the onehot vector, so the index of the non-zero value of each sample is saved in the inner list.
            // The outer list represents sequences contained in the batch.
            var inputBatch = new List<List<uint>>();
            // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
            var seqStartFlagBatch = new List<bool>();

            var inputSentences = new List<string>() { 
                "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                "BOS I want to book a flight from NewYork to Seattle EOS"
            };

            int numOfSequences = inputSentences.Count;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // The input for one sequence 
                seqData = new List<uint>();
                // Get the index of each word in the sentence.
                substring = inputSentences[seqIndex].Split(' ');
                foreach (var str in substring)
                {
                    var index = vocabToIndex[str];
                    seqData.Add(index);
                }
                inputBatch.Add(seqData);
                seqStartFlagBatch.Add(true);
            }

            // Create the Value representing the batch data.
            inputValue = Value.CreateBatchOfSequences<float>(vocabSize, inputBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);

            // Build input and output data map
            inputDataMap[inputVar] = inputValue;
            outputDataMap[outputVar] = null;

            // Evalaute the model
            myFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluation result.
            outputData = new List<List<uint>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyTo(outputVar, outputData);

            // output the result
            var numOfElementsInSample = vocabSize;
            uint seqNo = 0;
            foreach (var seq in outputData)
            {
                Console.Write("Seq=" + seqNo + ":");
                foreach (var index in seq)
                {
                    // get the word based on index
                    Console.Write(indexToVocab[index]);
                }
                Console.WriteLine();
                // next sequence.
                seqNo++;
            }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("======== Evaluate model using C# ========");

            EvaluationWithDenseData(DeviceDescriptor.CPUDevice);
            EvaluationWithOneHot(DeviceDescriptor.CPUDevice);

            // For using GPU:
            //EvaluationWithDenseData(DeviceDescriptor.GPUDevice(0));
            //EvaluationWithOneHot(DeviceDescriptor.GPUDevice(1));

            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
