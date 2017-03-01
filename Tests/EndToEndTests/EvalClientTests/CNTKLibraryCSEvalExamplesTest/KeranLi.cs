using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;
using System.Drawing;
using System.IO;

namespace CNTKLibraryCSEvalExamples
{
    class KeranLi
    {
        public static void EvaluationBatchOfImages(DeviceDescriptor device)
        {
            const string outputName = "p";
            // const string outputName = "z";
            const string inputName = "features";
            var inputDataMap = new Dictionary<Variable, Value>();

            // Load the model.
            Function rootFunc = Function.LoadModel(@"C:\CNTKMisc\KeranLi\OpensetSample\InceptionV3.0", device);
            //var outputFunc = rootFunc.FindByName(outputName);
            //var modelFunc = CNTKLib.AsComposite(outputFunc);
            var modelFunc = rootFunc;

            // Get output variable based on name
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.Where(variable => string.Equals(variable.Name, inputName)).Single();
            var outputDataMap = new Dictionary<Variable, Value>();
            Value inputVal, outputVal;
            List<List<float>> outputBuffer;

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            Console.WriteLine("\nEvaluate batch of images");

            Bitmap bmp, resized;
            List<float> resizedCHW;

            //var fileList = File.ReadAllLines("D:/Projects/CognitiveService/TestSet/TestMapSample_Kik_AdultRacy_20170215.txt").ToList();
            //fileList = fileList.Where(file => Math.Abs(file.GetHashCode() % 1000) == 0).ToList();
            var fileList = new List<string>() {  // @"C:\CNTKMisc\KeranLi\OpensetSample\1.jpg",
                @"C:\CNTKMisc\KeranLi\OpensetSample\2.jpg"
                // @"C:\CNTKMisc\KeranLi\OpensetSample\3.jpg"
                };
            var seqData = new List<float>();
            for (int sampleIndex = 0; sampleIndex < fileList.Count; sampleIndex++)
            {
                var filename = fileList[sampleIndex];
                if (!File.Exists(filename))
                {
                    throw new FileNotFoundException(string.Format("File '{0}' not found.", filename));
                }
                bmp = new Bitmap(Bitmap.FromFile(filename));
                resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
                resizedCHW = resized.ParallelExtractCHW();
                // Aadd this sample to the data buffer.
                seqData.AddRange(resizedCHW);
            }

            // Create Value for the batch data.
            inputVal = Value.CreateBatch(inputVar.Shape, seqData, device);

            // Create input data map.
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Evaluate the model against the batch input
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Retrieve the evaluation result.
            outputBuffer = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputBuffer);

            // Output result
            
            PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
        }

        private static void PrintOutput<T>(uint sampleSize, List<List<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            uint outputSampleSize = sampleSize;
            foreach (var seq in outputBuffer)
            {
                if (seq.Count % outputSampleSize != 0)
                {
                    throw new ApplicationException("The number of elements in the sequence is not a multiple of sample size");
                }

                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                uint i = 0;
                uint sampleNo = 0;
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

    }
}
