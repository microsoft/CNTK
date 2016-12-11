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
    //
    // This example shows how to use the Sequence class to prepare input and output.
    //
    public class SequenceClassesSample
    {
        // The example shows how to use Sequence class.
        public static void DenseSequence()
        {
            // Load the model.
            Function modelFunc = Function.LoadModel("z.model");

            Variable outputVar = modelFunc.Output;
            Variable inputVar = modelFunc.Arguments.Single();

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            // Number of sequences for this batch
            int numOfSequences = 2;
            // Number of samples in each sequence
            int[] numOfSamplesInSequence = { 3, 3 };

            // inputData contains mutliple sequences. Each sequence has multiple samples.
            var inputBatch = new List<Sequence<float>>();
            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png", "00005.png" };
            int fileIndex = 0;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Create a new data buffer for the new sequence
                var seqData = new Sequence<float>(inputVar.Shape);
                for (int sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    // Aadd this sample to the data buffer of this sequence
                    seqData.AddRange(resizedCHW);
                }
                // Add this sequence to the sequences list
                inputBatch.Add(seqData);
            }

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            // void Create<T>(Shape shape, List<List<T>> data, DeviceDescriptor computeDevice)
            inputMap.Add(inputVar, Value.Create(inputBatch, DeviceDescriptor.CPUDevice));

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            // Use Variables in input and output maps.
            // It is also possible to use variable name in input and output maps.
            modelFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            var outputData = new List<Sequence<float>>();
            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            // void CopyTo(Variable, List<Sequence<T>>)
            outputVal.CopyTo(outputVar, outputData);

            // Output results
            // Use sample based iterator to get result.
        }

        // 
        // The example uses the SequenceOneHot class as input and output for evaluation
        // The input data contains multiple sequences and each sequence contains multiple samples.
        // There is only one non-zero value in each sample, so the sample can be represented by the index of this non-zero value
        //
        public static void OneHotSequence()
        {
            var vocabToIndex = new Dictionary<string, uint>();
            var indexToVocab = new Dictionary<uint, string>();

            Function myFunc = Function.LoadModel("atis.model");

            // Get input variable 
            const string inputNodeName = "features";
            var inputVar = myFunc.Arguments.Where(variable => string.Equals(variable.Name, inputNodeName)).Single();

            uint vocabSize = inputVar.Shape.TotalSize;

            // The input data. 
            // Each sample is represented by a onehot vector, so the index of the non-zero value of each sample is saved in the inner list
            // The outer list represents sequences of the batch.
            var inputBatch = new List<SequenceOneHotVector>();
            var inputSentences = new List<string>() { 
                "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
                "BOS I want to book a flight from NewYork to Seattle EOS"
            };

            // The number of sequences in this batch
            int numOfSequences = inputSentences.Count;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // the input for one sequence 
                var seqData = new SequenceOneHotVector(vocabSize);
                // Get the word from the sentence.
                string[] substring = inputSentences[seqIndex].Split(' ');
                foreach (var str in substring)
                {
                    // Get the index of the word
                    var index = vocabToIndex[str];
                    // Add the sample to the sequence
                    seqData.Add(index);
                }
                // Add the sequence to the batch
                inputBatch.Add(seqData);
            }

            // Create the Value representing the data.
            // void CreateValue<T>(List<SequenceOneHotVector>, DeviceDescriptor computeDevice) 
            Value inputValue = Value.Create<float>(inputBatch, DeviceDescriptor.CPUDevice);

            // Create input map
            var inputMap = new Dictionary<Variable, Value>();
            inputMap.Add(inputVar, inputValue);

            // Prepare output
            const string outputNodeName = "out.z_output";
            Variable outputVar = myFunc.Outputs.Where(variable => string.Equals(variable.Name, outputNodeName)).Single();

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<Variable, Value>();
            outputMap.Add(outputVar, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            myFunc.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            var outputData = new List<SequenceOneHotVector>();
            Value outputVal = outputMap[outputVar];

            // Get output as onehot vector
            // void CopyTo(List<SequenceOneHotVector>)
            outputVal.CopyTo<float>(outputVar, outputData);
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
        // The example uses SequenceSparse to prepare input and output data for evaluation
        // The input data contains multiple sequences and each sequence contains multiple samples.
        // Each sample is a n-dimensional tensor. For sparse input, the n-dimensional tensor needs 
        // to be flatted into 1-dimensional vector and the index of non-zero values in the 1-dimensional vector
        // is used as sparse input.
        //
        static void SparseSequence()
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

            // The input batch having sparse input.
            var inputBatch = new List<SequenceSparse<float>>();
         
            // Assuming the images to be evlauated are quite sparse so using sparse input is a better option than dense input.
            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png", "00005.png" };
            int fileIndex = 0;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                var seq = new SequenceSparse<float>(inputShape);

                // Input data for the sequence
                var dataList = new List<float>();
                var indexList = new List<uint>();
                for (int sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    // For any n-dimensional tensor, it needs first to be flatted to 1-dimension.
                    // The index in the sparse input refers to the position in the flatted 1-dimension vector.
                    uint index = 0;
                    foreach (var v in resizedCHW)
                    {
                        // Put non-zero value into data
                        // put the index of this value into indexList
                        if (v != 0)
                        {
                            dataList.Add(v);
                            indexList.Add(index);
                        }
                        index++;
                    }
                    // Add the sample to the sequence
                    seq.AddSample(dataList, indexList);
                }
                // Add this sequence to the input
                inputBatch.Add(seq);
            }

            // Create value object from data.
            // void Create<T>(List<SequenceSparse<T>> data, DeviceDescriptor computeDevice) 
            Value inputValue = Value.Create<float>(inputBatch, DeviceDescriptor.CPUDevice);

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
            var outputData = new List<SequenceSparse<float>>();

            Value outputVal = outputMap[outputVar];
            // Get output result as dense output
            // CopyTo(Variable, List<SequenceSparse<T>> data)
            outputVal.CopyTo(outputVar, outputData);

            // Output results
            // provide sample-based iterator
        }
    }
}
