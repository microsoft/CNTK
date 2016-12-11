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
    public class EvalClassSample
    {
        public static void EvaluateUsingCSEvalLib()
        {
            // Load the model
            var model = new CNTK.Evaluation();

            model.LoadModel("z.model", DeviceDescriptor.CPUDevice);

            const string outputNodeName = "Plus2060_output";
            // The model has empty node name.
            // Todo: define the name for the input node, or use SetEvaluationOutput to get related input variables.
            const string inputNodeName = "";

            // Get shape data for the input variable
            var inputDims = model.InputsDimensions[inputNodeName];
            // Todo: add property to Shape
            uint imageWidth = inputDims[0];
            uint imageHeight = inputDims[1];
            uint imageChannels = inputDims[2];
            uint imageSize = model.InputsSize[inputNodeName];

            // Number of sequences for this batch
            int numOfSequences = 2;
            // Number of samples in each sequence
            int[] numOfSamplesInSequence = { 3, 3 };

            // inputData contains mutliple sequences. Each sequence has multiple samples.
            // Each sample has the same tensor shape.
            // The outer List is the sequences. Its size is numOfSequences.
            // The inner List is the inputs for one sequence. Its size is inputShape.TotalSize * numberOfSampelsInSequence
            var inputData = new List<List<float>>();
            var fileList = new List<string>() { "00000.png", "00001.png", "00002.png", "00003.png", "00004.png", "00005.png" };
            int fileIndex = 0;
            for (int seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                // Create a new data buffer for the new sequence
                var seqData = new List<float>();
                for (int sampleIndex = 0; sampleIndex < numOfSamplesInSequence[seqIndex]; sampleIndex++)
                {
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(fileList[fileIndex++]));
                    var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                    List<float> resizedCHW = resized.ParallelExtractCHW();
                    // Aadd this sample to the data buffer of this sequence
                    seqData.AddRange(resizedCHW);
                }
                // Add this sequence to the sequences list
                inputData.Add(seqData);
            }

            // Create input map
            var inputMap = new Dictionary<string, Value>();
            // void Create<T>(Shape shape, List<List<T>> data, DeviceDescriptor computeDevice)
            inputMap.Add(inputNodeName, model.CreateValue<float>(inputNodeName, inputData, DeviceDescriptor.CPUDevice));

            // Create ouput map. Using null as Value to indicate using system allocated memory.
            var outputMap = new Dictionary<string, Value>();
            outputMap.Add(outputNodeName, null);

            // Evalaute
            // Todo: test on GPUDevice()?
            model.Evaluate(inputMap, outputMap, DeviceDescriptor.CPUDevice);

            // The buffer for storing output for this batch
            var outputData = new List<List<float>>();
            Value outputVal = outputMap[outputNodeName];
            // Get output result as dense output
            // void CopyTo(List<List<T>>
            model.CopyValueTo<float>(outputNodeName, outputVal, outputData);
        }
    }
}
