using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using CNTK;
using Amoc.Core;
namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {

            try
            {
                string modelFilePath = "FCN2.094";
                ThrowIfFileNotExist(modelFilePath, string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/Classification/ResNet to create the model.", modelFilePath));

                //this is the crashed location
                Function modelFunc = Function.LoadModel(modelFilePath, DeviceDescriptor.CPUDevice);

                Console.WriteLine("\n===== Evaluate single image =====");
                foreach (var file in new List<string> { "F04_423C020A.mfc", "F04_423C020N.mfc" })
                {
                    //this is the correct location
                    //Function modelFunc = Function.LoadModel(modelFilePath, DeviceDescriptor.CPUDevice);
                    Variable outputVar = modelFunc.Output;
                    Variable inputVar = modelFunc.Arguments.Single();

                    //HtkFeatureFile htkFile = HtkFeatureFile.Read(new MemoryStream(row[inputRow].Binary));
                    HtkFeatureFile htkFile = HtkFeatureFile.Read(file);
                    int featDim = htkFile.BytesPerSample / sizeof(float);
                    int context = (Convert.ToInt32(inputVar.Shape[0]) / featDim - 1) / 2;
                    List<List<float>> frms = htkFile.Frames.Select(x => x.ToList()).ToList();
                    var data = new List<float>();
                    foreach (var frm in ContextFrames(frms, context))
                    {
                        data.AddRange(frm);
                    }


                    //Value outputVal;
                    var outputBuffer = new List<List<float>>();

                    var inputDataMap = new Dictionary<Variable, Value> { { inputVar, Value.CreateBatch(inputVar.Shape, data, DeviceDescriptor.CPUDevice) } };
                    var outputDataMap = new Dictionary<Variable, Value> { { outputVar, null } };

                    modelFunc.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);

                    outputDataMap[outputVar].CopyVariableValueTo(outputVar, outputBuffer);

                    PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }
        public static List<List<float>> ContextFrames(List<List<float>> frames, int context)
        {
            var extented = new List<List<float>>();
            for (int i = 0; i < frames.Count; i++)
            {
                var item = new List<float>();

                for (int j = -context; j <= context; j++)
                {
                    if (i + j < 0)
                        item.AddRange(frames.First());
                    else if (i + j >= frames.Count)
                        item.AddRange(frames.Last());
                    else
                        item.AddRange(frames[i + j]);
                }
                extented.Add(item);
            }
            return extented;
        }

        /// <summary>
        /// Print out the evalaution results.
        /// </summary>
        /// <typeparam name="T">The data value type</typeparam>
        /// <param name="sampleSize">The size of each sample.</param>
        /// <param name="outputBuffer">The evaluation result data.</param>
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
        /// <summary>
        /// Checks whether the file exists. If not, write the error message on the console and throw FileNotFoundException.
        /// </summary>
        /// <param name="filePath">The file to check.</param>
        /// <param name="errorMsg">The message to write on console if the file does not exist.</param>
        private static void ThrowIfFileNotExist(string filePath, string errorMsg)
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


    }
}
