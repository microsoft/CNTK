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
using Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient;

namespace CSEvalV2Example
{
    public class Program
    {
        // 
        // The example shows 
        // - how to load model.
        // - how to set desired ouput variables for evaluation and retrieve required input variables.
        // - how to prepear input data (a batch with varaible sequences) in dense format, using List<List<T>>.
        // - how to evaluate a model
        // - how to get out output data in dense format, using List<List<T>>.
        //
        static void DenseExample()
        {
            const string outputNodeName = "Plus2060_output";
            // The model has an empty input node name. Fortunatelly there is only one input node for the model.
            const string inputNodeName = "";

            // Load the model.
            Function modelFunc = Function.LoadModel("z.model");

            // Todo: how to get a variable in the intermeidate layer by name?
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();
          
            // Set desired output variables and get required inputVariables;
            Function evalFunc = Function.Combine(new Variable[] {outputVar});
            var inputVarList = evalFunc.Arguments;

            Variable inputVar = inputVarList.Where(variable => string.Equals(variable.Name, inputNodeName)).Single();

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;
            var inputMap = new Dictionary<Variable, Value>();
            var outputMap = new Dictionary<Variable, Value>();

            // Evaluate with single image
            Console.WriteLine("Evaluate single image");

            Bitmap bmp = new Bitmap(Bitmap.FromFile("00000.png"));
            var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            var inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, DeviceDescriptor.CPUDevice);
            // Create input map
            // Create the Value from input data and add to the input map.
            // void Create<T>(NDShape shape, List<List<T>> data, DeviceDescriptor computeDevice)
            inputMap.Add(inputVar, inputVal);

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            outputMap.Add(outputVar, null);

            evalFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // The buffer for storing output for this batch
            var outputData = new List<List<float>>();
            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            // void CopyTo(Variable, List<List<T>>
            outputVal.CopyTo(outputVar, outputData);

            // Evaluate with batch of images
            Console.WriteLine("Evaluate batch of images");

            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png" };
            var seqData = new List<float>();
            for (int sampleIndex = 0; sampleIndex < fileList.Count; sampleIndex++)
            {
                bmp = new Bitmap(Bitmap.FromFile(fileList[sampleIndex++]));
                resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                resizedCHW = resized.ParallelExtractCHW();
                // Aadd this sample to the data buffer of this sequence
                seqData.AddRange(resizedCHW);
            }

            inputVal = Value.CreateBatch(inputVar.Shape, seqData, DeviceDescriptor.CPUDevice);
            inputMap[inputVar] = inputVal;
            outputMap[outputVar] = null;
            evalFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // Evaluate with sequence of images
            Console.WriteLine("Evaluate sequence of images");

            fileList = new List<string>() { "00000.png", "00001.png", "00002.png" };
            seqData = new List<float>();
            bool seqStartFlag = true;
            for (int sampleIndex = 0; sampleIndex < fileList.Count; sampleIndex++)
            {
                bmp = new Bitmap(Bitmap.FromFile(fileList[sampleIndex++]));
                resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                resizedCHW = resized.ParallelExtractCHW();
                // Aadd this sample to the data buffer of this sequence
                seqData.AddRange(resizedCHW);
            }

            inputMap[inputVar] = Value.CreateSequence(inputVar.Shape, seqData, seqStartFlag, DeviceDescriptor.CPUDevice);
            outputMap[outputVar] = null;
            evalFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            outputData = new List<List<float>>();
            outputVal = outputMap[outputVar];
            // Get output result as dense output
            // void CopyTo(Variable, List<List<T>>
            outputVal.CopyTo(outputVar, outputData);

            // Evalaute with batch of sequences 
            Console.WriteLine("Evaluate batch of sequences of images");

            // Number of sequences for this batch
            int numOfSequences = 2;
            // Number of samples in each sequence
            int[] numOfSamplesInSequence = { 3, 3 };

            // inputData contains mutliple sequences. Each sequence has multiple samples.
            // Each sample has the same tensor shape.
            // The outer List is the sequences. Its size is the number of qequences.
            // The inner List is the samples of one single sequence. Its size is inputShape.TotalSize * numberOfSampelsInSequence
            var inputBatch = new List<List<float>>();
            var seqStartFlagBatch = new List<bool>();

            fileList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png", "00005.png" };
            int fileIndex = 0;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Create a new data buffer for the new sequence
                seqData = new List<float>();
                for (int sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    resizedCHW = resized.ParallelExtractCHW();
                    // Aadd this sample to the data buffer of this sequence
                    seqData.AddRange(resizedCHW);
                }
                // Add this sequence to the sequences list
                inputBatch.Add(seqData);
                seqStartFlagBatch.Add(true);
            }

            // Create the Value from input data and add to the input map.
            // void Create<T>(NDShape shape, List<List<T>> data, DeviceDescriptor computeDevice)
            inputMap[inputVar] = Value.CreateBatchOfSequences(inputVar.Shape, inputBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);

            outputMap[outputVar] = null;
            evalFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            outputData = new List<List<float>>();
            outputVal = outputMap[outputVar];
            // Get output result as dense output
            // void CopyTo(Variable, List<List<T>>
            outputVal.CopyTo(outputVar, outputData);
            
            // Output results
            // Todo: add sample based iterator
            var numOfElementsInSample = outputVar.Shape.TotalSize;
            uint seqNo = 0;
            foreach (var seq in outputData)
            {
                uint elementIndex = 0;
                uint sampleIndex = 0;
                foreach (var data in seq)
                {
                    // a new sample starts.
                    if (elementIndex++ == 0)
                    {
                        Console.Write("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    }
                    Console.Write(" " + data);
                    // reach the end of a sample.
                    if (elementIndex == numOfElementsInSample)
                    {
                        Console.WriteLine(".");
                        elementIndex = 0;
                        sampleIndex++;
                    }
                }
                seqNo++;
            }
        }

        // 
        // The example shows 
        // - how to use OneHot vector as input and output for evaluation
        //   The input data contains multiple sequences and each sequence contains multiple samples.
        //   There is only one non-zero value in each sample, so the sample can be represented by the index of this non-zero value
        // - use variable name, instead of Variable, for as parameters for evaluate.
        //
        static void OneHotExample()
        {
            var vocabToIndex = new Dictionary<string, uint>();
            var indexToVocab = new Dictionary<uint, string>();

            Function myFunc = Function.LoadModel("atis.model");

            // Get input variable 
            const string inputNodeName = "features";
            var inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputNodeName)).Single();

            uint vocabSize = inputVar.Shape.TotalSize;

            // Evalaute a single sequence
            var inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
            // the input for one sequence 
            var seqData = new List<uint>();
            var seqStartFlag = true;
            // Get the word from the sentence.
            string[] substring = inputSentence.Split(' ');
            foreach (var str in substring)
            {
                // Get the index of the word
                var index = vocabToIndex[str];
                // Add the sample to the sequence
                seqData.Add(index);
            }

            // Create input map
            var inputValue = Value.CreateSequence<float>(vocabSize, seqData, seqStartFlag, DeviceDescriptor.CPUDevice);
            var inputMap = new Dictionary<string, Value>();
            inputMap.Add(inputNodeName, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<string, Value>();
            outputMap.Add(outputNodeName, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            var outputData = new List<List<uint>>();
            Value outputVal = outputMap[outputNodeName];

            // Get output as onehot vector
            // void CopyTo(Variable, List<List<uint>>)
            outputVal.CopyTo(outputVar, outputData);

            // Evaluate batch of sequences.

            // The input data. 
            // Each sample is represented by a onehot vector, so the index of the non-zero value of each sample is saved in the inner list
            // The outer list represents sequences of the batch.
            var inputBatch = new List<List<uint>>();
            var seqStartFlagBatch = new List<bool>();
            var inputSentences = new List<string>() { 
                "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                "BOS I want to book a flight from NewYork to Seattle EOS"
            };

            // The number of sequences in this batch
            int numOfSequences = inputSentences.Count;

            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // the input for one sequence 
                seqData = new List<uint>();
                // Get the word from the sentence.
                substring = inputSentences[seqIndex].Split(' ');
                foreach (var str in substring)
                {
                    // Get the index of the word
                    var index = vocabToIndex[str];
                    // Add the sample to the sequence
                    seqData.Add(index);
                }
                // Add the sequence to the batch
                inputBatch.Add(seqData);
                seqStartFlagBatch.Add(true);
            }

            // Create the Value representing the data.
            // void CreateValue<T>(uint vocabularySize, List<List<uint> oneHotIndexes, DeviceDescriptor computeDevice) 
            inputValue = Value.CreateBatchOfSequences<float>(vocabSize, inputBatch, seqStartFlagBatch, DeviceDescriptor.CPUDevice);
            inputMap[inputNodeName] = inputValue;

            outputMap[outputNodeName] = null;

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            outputData = new List<List<uint>>();
            outputVal = outputMap[outputNodeName];

            // Get output as onehot vector
            // void CopyTo(Variable, List<List<uint>>)
            outputVal.CopyTo(outputVar, outputData);
            var numOfElementsInSample = vocabSize;

            // output the result
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

        // 
        // The example uses sparse input and output for evaluation
        // The input data contains multiple sequences and each sequence contains multiple samples.
        // Each sample is a n-dimensional tensor. For sparse input, the n-dimensional tensor needs 
        // to be flatted into 1-dimensional vector and the index of non-zero values in the 1-dimensional vector
        // is used as sparse input.
        //
        static void SparseExample()
        {
            // Load the model.
            Function myFunc = Function.LoadModel("resnet.model");

            // Get the input variable from by name
            const string inputNodeName = "features";
            // Todo: provide a help method in Function: getVariableByName()? Or has a property variables which is dictionary of <string, Variable>
            Variable inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputNodeName)).Single();

            // Get shape data 
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            // Number of sequences for this batch
            int numOfSequences = 2;
            // Number of samples in each sequence
            int[] numOfSamplesInSequence = { 4, 2 };

            // inputData contains all inputs for the evaluation
            // The inner List is the inputs for one sequence. Its size is inputShape.TotalSize() * numberOfSampelsInSequence
            // The outer List is the sequences. Its size is numOfSequences; 
            var dataOfSequences = new List<List<float>>();
            var indexOfSequences = new List<List<uint>>();
            var nnzCountOfSequences = new List<List<uint>>();
            var seqStartFlagBatch = new List<bool>();
            // Assuming the images to be evlauated are quite sparse so using sparse input is a better option than dense input.
            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png", "00005.png" };
            int fileIndex = 0;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Input data for the sequence
                var dataList = new List<float>();
                var indexList = new List<uint>();
                var nnzCountList = new List<uint>();
                for (int sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    // For any n-dimensional tensor, it needs first to be flatted to 1-dimension.
                    // The index in the sparse input refers to the position in the flatted 1-dimension vector.
                    uint nnzCount = 0;
                    uint index = 0;
                    foreach (var v in resizedCHW)
                    {
                        // Put non-zero value into data
                        // put the index of this value into indexList
                        if (v != 0)
                        {
                            dataList.Add(v);
                            indexList.Add(index);
                            nnzCount++;
                        }
                        index++;
                    }
                    // Add nnzCount of this sample to nnzCountList
                    nnzCountList.Add(nnzCount);
                }
                // Add this sequence to the list
                dataOfSequences.Add(dataList);
                indexOfSequences.Add(indexList);
                nnzCountOfSequences.Add(nnzCountList);
                seqStartFlagBatch.Add(true);
            }

            // Create value object from data.
            // void Create<T>(Shape shape, List<List<T>> data, List<List<uint> indexes, List<List<uint>> nnzCounts, DeviceDescriptor computeDevice) 
            Value inputValue = Value.CreateBatchOfSequences<float>(inputVar.Shape, dataOfSequences, indexOfSequences, nnzCountOfSequences, seqStartFlagBatch, DeviceDescriptor.CPUDevice);

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            inputMap.Add(inputVar, inputValue);

            // Repeat the steps above for each input.

            // Prepare output
            const string outputNodeName = "out.z_output";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // The buffer for storing output for this batch
            var outputData = new List<List<float>>();
            var outputIndex = new List<List<uint>>();
            var outputNnzCount = new List<List<uint>>();

            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            // CopyTo(List<List<T>> data, List<List<uint> indexes, List<List<uint>> nnzCounts)
            outputVal.CopyTo(outputVar, outputData, outputIndex, outputNnzCount);
            var outputShape = outputVar.Shape;

            // Output results
            var numOfElementsInSample = outputVar.Shape.TotalSize;
            uint seqNo = 0;
            for (int seqIndex = 0; seqIndex < outputData.Count; seqIndex++)
            {
                var dataList = outputData[seqIndex];
                var indexList = outputIndex[seqIndex];
                var nnzCountList = outputIndex[seqIndex];
                int index = 0;
                for (int sampleIndex = 0; sampleIndex < nnzCountList.Count; sampleIndex++)
                {
                    int elementIndex = 0;
                    Console.WriteLine("Seq=" + seqNo + ", Sample=" + sampleIndex + ":");
                    Console.Write("   Element " + elementIndex + ":");
                    for (int c = 0; c < nnzCountList[c]; c++)
                    {
                        Console.Write("[" + indexList[index] + "]=" + dataList[index] + ", ");
                        index++;
                    }
                    elementIndex++;
                    Console.WriteLine(".");
                }
                seqNo++;
            }
        }

        static void Main(string[] args)
        {
            Console.WriteLine("======== Evaluate model using C# ========");
            DenseExample();

            //Console.WriteLine("======== Evaluate Using EvalV2Library ========");
            //EvalClassSample.EvaluateUsingCSEvalLib();

            //Console.WriteLine("======== Evaluate V1 Model ========");
            //SampleUsingBasicClass.EvaluateV1ModelUsingNDView();
            //Console.WriteLine("======== Evaluate V2 Model ========");
            //SampleUsingBasicClass.EvaluateV2ModelUsingNDView();
            //Console.WriteLine("======== Evaluate Model Using System Allocated Memory for Output Value ========");
            //SampleUsingBasicClass.EvaluateUsingSystemAllocatedMemory();
            //Console.WriteLine("======== Evaluate Using Value::Create ========");
            //SampleUsingBasicClass.EvaluateUsingCreateValue();

            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
