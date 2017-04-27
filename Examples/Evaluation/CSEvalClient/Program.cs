//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs -- main C# file that contains client code to call the CLI Wrapper class.
//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Threading;
using System.Threading.Tasks;
using CNTKLibraryCSEvalExamples;

namespace Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient
{
    /// <summary>
    /// Program for demonstrating how to run model evaluations using the CLIWrapper
    /// </summary>
    /// <description>
    /// This program is a managed client using the CLIWrapper to run the model evaluator in CNTK.
    /// There are four cases shown in this program related to model loading, network creation and evaluation.
    /// 
    /// To run this program from the CNTK binary drop, you must add the NuGet package for model evaluation first.
    /// Refer to <see cref="https://github.com/Microsoft/CNTK/wiki/NuGet-Package"/> for information regarding the NuGet package for model evaluation.
    /// 
    /// EvaluateModelSingleLayer and EvaluateModelMultipleLayers
    /// --------------------------------------------------------
    /// These two cases require the 01_OneHidden model which is part of the <CNTK>/Examples/Image/GettingStarted example.
    /// Refer to <see cref="https://github.com/Microsoft/CNTK/blob/master/Examples/Image/GettingStarted/README.md"/> for how to train
    /// the model used in these examples.
    /// 
    /// EvaluateNetworkSingleLayer and EvaluateNetworkSingleLayerNoInput
    /// ----------------------------------------------------------------
    /// These two cases do not required a trained model (just the network description). These cases show how to extract values from a single forward-pass
    /// without any input to the model.
    /// 
    /// EvaluateMultipleModels
    /// ----------------------
    /// This case requires the 02_Convolution model and the Test-28x28_cntk_text.txt test file which are part of the <CNTK>/Examples/Image/GettingStarted example.
    /// Refer to <see cref="https://github.com/Microsoft/CNTK/blob/master/Examples/Image/GettingStarted/README.md"/> for how to train
    /// the model used in this example.
    /// 
    /// EvaluateImageClassificationModel
    /// -----------------------
    /// This case requires the ResNet_18 trained model which can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_18.model"/>.
    /// This case shows how to evaluate a model that was trained with the ImageReader.
    /// The input for evaluation needs to be transformed in a similar manner as the ImageReader did during training.
    /// 
    /// </description>
    class Program
    {
        private static string initialDirectory;
        /// The location of the Resnet model file that is required for the image API tests.
        private static string resnetModelFilePath;
        /// The location of the test image that is using in the image API tests.
        private static string imageFileName;
        // The width and height of the images that go into ResNet.
        private static int resNetImageSize = 224;


        /// <summary>
        /// Program entry point
        /// </summary>
        /// <param name="args">Program arguments (ignored)</param>
        private static void Main(string[] args)
        {
            initialDirectory = Environment.CurrentDirectory;

            Console.WriteLine("====== EvaluateModelSingleLayer ========");
            EvaluateModelSingleLayer();

            Console.WriteLine("\n====== EvaluateModelMultipleLayers ========");
            EvaluateModelMultipleLayers();

            Console.WriteLine("\n====== EvaluateExtendedNetworkSingleLayerNoInput ========");
            EvaluateExtendedNetworkSingleLayerNoInput();

            Console.WriteLine("\n====== EvaluateMultipleModels ========");
            EvaluateMultipleModels();

            // The image tests require the Resnet model. 
            // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_18.model"/>
            // The model is assumed to be located at: <CNTK>\Examples\Image\Classification\ResNet 
            // along with a sample image file named "zebra.jpg".
            var resnetDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Image\Classification\ResNet");
            resnetModelFilePath = Path.Combine(resnetDirectory, "ResNet_18.model");
            ThrowIfFileNotExist(resnetModelFilePath,
                string.Format("Error: The model '{0}' does not exist. Please download the model from https://www.cntk.ai/resnet/ResNet_18.model and save it under ..\\..\\Examples\\Image\\Classification\\ResNet.", resnetModelFilePath));
            imageFileName = Path.Combine(resnetDirectory, "zebra.jpg");
            ThrowIfFileNotExist(imageFileName, string.Format("Error: The test image file '{0}' does not exist.", imageFileName));

            Console.WriteLine("\n====== EvaluateImageInputUsingFeatureVector ========");
            var outputs1 = EvaluateImageInputUsingFeatureVector();

            Console.WriteLine("\n====== EvaluateImageInputUsingImageApi ========");
            var outputs2 = EvaluateImageInputUsingImageApi();

            Console.WriteLine("\n====== CompareImageApiResults ========");
            CompareImageApiResults(outputs1, outputs2);

            // This pattern is used by End2EndTests to check whether the program runs to complete.
            Console.WriteLine("\n====== Evaluation Complete ========");
        }

        private static void CompareImageApiResults(List<float> outputs1,List<float> outputs2)
        {
            if (outputs1.Count != outputs2.Count)
            {
                throw new Exception("Both APIs must return the same number of output values.");
            }
            foreach (var i in Enumerable.Range(0, outputs1.Count))
            {
                if (Math.Abs(outputs1[i] - outputs2[i]) > 1e-5f)
                {
                    throw new Exception(String.Format("Output value mismatch at position {0}", i));
                }
            }
            Console.WriteLine("Both image API calls returned the same output vector.");
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

        /// <summary>
        /// Handle CNTK exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnCNTKException(CNTKException ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            throw ex;
        }

        /// <summary>
        /// Handle general exceptions.
        /// </summary>
        /// <param name="ex">The exception to be handled.</param>
        private static void OnGeneralException(Exception ex)
        {
            // The pattern "Inner Exception" is used by End2EndTests to catch test failure.
            Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            throw ex;
        }

        /// <summary>
        /// Evaluates a trained model and obtains a single layer output
        /// </summary>
        /// <remarks>
        /// This example requires the 01_OneHidden trained model
        /// </remarks>
        private static void EvaluateModelSingleLayer()
        {
            try
            {
                string outputLayerName;

                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                Environment.CurrentDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Image\GettingStarted");
                List<float> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Load model
                    string modelFilePath = Path.Combine(Environment.CurrentDirectory, @".\Output\Models\01_OneHidden");
                    ThrowIfFileNotExist(modelFilePath, 
                        string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/GettingStarted to create the model.", modelFilePath));

                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelFilePath), deviceId: -1);

                    // Generate random input values in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    var inputs = GetDictionary(inDims.First().Key, inDims.First().Value, 255);

                    // We request the output layer names(s) and dimension, we'll use the first one.
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);
                    outputLayerName = outDims.First().Key;
                    // We can call the evaluate method and get back the results (single layer)...
                    outputs = model.Evaluate(inputs, outputLayerName);
                }

                OutputResults(outputLayerName, outputs);
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }
        }

        /// <summary>
        /// Evaluates a trained model and obtains multiple layers output (including hidden layer)
        /// </summary>
        /// <remarks>
        /// This example requires the 01_OneHidden trained model
        /// </remarks>
        private static void EvaluateModelMultipleLayers()
        {
            try
            {
                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                Environment.CurrentDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Image\GettingStarted");

                Dictionary<string, List<float>> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Desired output layers
                    const string hiddenLayerName = "out.h1";
                    const string outputLayerName = "out.z";

                    // Load model
                    string modelFilePath = Path.Combine(Environment.CurrentDirectory, @".\Output\Models\01_OneHidden");
                    ThrowIfFileNotExist(modelFilePath,
                        string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/GettingStarted to create the model.", modelFilePath));

                    var desiredOutputLayers = new List<string>() { hiddenLayerName, outputLayerName };
                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelFilePath), deviceId: -1, outputNodeNames: desiredOutputLayers);

                    // Generate random input values in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    var inputs = GetDictionary(inDims.First().Key, inDims.First().Value, 255);

                    // We request the output layer names(s) and dimension, we'll get both the hidden layer and the output layer
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);

                    // We can preallocate the output structure and pass it in (multiple output layers)
                    outputs = new Dictionary<string, List<float>>()
                    {
                        { hiddenLayerName, GetFloatArray(outDims[hiddenLayerName], 1) },    
                        { outputLayerName, GetFloatArray(outDims[outputLayerName], 1) }
                    };
                    model.Evaluate(inputs, outputs);
                }

                OutputResults(outputs);
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }
        }

        /// <summary>
        /// Evaluates an extended network (without a model and without input) and obtains a single layer output
        /// </summary>
        private static void EvaluateExtendedNetworkSingleLayerNoInput()
        {
            const string modelDefinition = @"precision = ""float"" 
                                     traceLevel = 1
                                     run=NDLNetworkBuilder
                                     NDLNetworkBuilder=[
                                     v1 = Constant(1)
                                     v2 = Constant(2, tag=""output"")
                                     ol = Plus(v1, v2, tag=""output"")
                                     FeatureNodes = (v1)
                                     ]";

            try
            {
                using (var model = new ModelEvaluationExtendedF())
                {
                    // Create the network
                    model.CreateNetwork(modelDefinition);

                    VariableSchema outputSchema = model.GetOutputSchema();

                    var outputNodeNames = outputSchema.Select(s => s.Name).ToList<string>();
                    model.StartForwardEvaluation(outputNodeNames);

                    var outputBuffer = outputSchema.CreateBuffers<float>();
                    var inputBuffer = new ValueBuffer<float>[0];

                    // We can call the evaluate method and get back the results...
                    model.ForwardPass(inputBuffer, outputBuffer);

                    // We expect two outputs: the v2 constant, and the ol Plus result
                    var expected = new float[][] { new float[] { 2 }, new float[] { 3 } };

                    Console.WriteLine("Expected values: {0}", string.Join(" - ", expected.Select(b => string.Join(", ", b)).ToList<string>()));
                    Console.WriteLine("Actual Values  : {0}", string.Join(" - ", outputBuffer.Select(b => string.Join(", ", b.Buffer)).ToList<string>()));
                }
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }
        }

        /// <summary>
        /// Evaluates multiple instances of a model in the same process.
        /// </summary>
        /// <remarks>
        /// Although all models execute concurrently (multiple tasks), each model is evaluated with a single task at a time.
        /// </remarks>
        private static void EvaluateMultipleModels()
        {
            // Specifies the number of models in memory as well as the number of parallel tasks feeding these models (1 to 1)
            int numConcurrentModels = 4;

            // Specifies the number of times to iterate through the test file (epochs)
            int numRounds = 1;

            // Counts the number of evaluations accross all models
            int count = 0;

            // Counts the number of failed evaluations (output != expected) accross all models
            int errorCount = 0;

            // The examples assume the executable is running from the data folder
            // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
            Environment.CurrentDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Image\GettingStarted");

            // Load model
            string modelFilePath = Path.Combine(Environment.CurrentDirectory, @".\Output\Models\02_OneConv");
            ThrowIfFileNotExist(modelFilePath, 
                string.Format("Error: The model '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/GettingStarted to create the model.", modelFilePath));

            // Initializes the model instances
            ModelEvaluator.Initialize(numConcurrentModels, modelFilePath);

            string testfile = Path.Combine(Environment.CurrentDirectory, @"..\DataSets\MNIST\Test-28x28_cntk_text.txt");
            ThrowIfFileNotExist(testfile, 
                string.Format("Error: The test file '{0}' does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/GettingStarted to download the data.", testfile));

            Stopwatch sw = new Stopwatch();
            sw.Start();

            try
            {
                for (int i = 0; i < numRounds; i++)
                {
                    // Feed each line to a single model in parallel
                    Parallel.ForEach(File.ReadLines(testfile), new ParallelOptions() { MaxDegreeOfParallelism = numConcurrentModels }, (line) =>
                    {
                        Interlocked.Increment(ref count);

                        // The file format correspond to the CNTK Text Format Reader format (https://github.com/Microsoft/CNTK/wiki/CNTKTextFormat-Reader)
                        var sets = line.Split('|');
                        var labels = sets[1].Trim().Split(' ').Skip(1);
                        var features = sets[2].Trim().Split(' ').Skip(1);

                        // Retrieve the 1-hot vector with the label index
                        var expected = labels.Select(float.Parse).Select((v, index) => new { Value = v, Index = index })
                            .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                            .Index;

                        // Retrieve the features
                        var inputs = features.Select(float.Parse).ToList();

                        // We can call the evaluate method and get back the results (single layer)...
                        var outputs = ModelEvaluator.Evaluate(inputs);

                        // Retrieve the outcome index (so we can compare it with the expected index)
                        var max = outputs.Select((v, index) => new { Value = v, Index = index })
                            .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                            .Index;

                        // Count the errors
                        if (expected != max)
                        {
                            Interlocked.Increment(ref errorCount);
                        }
                    });
                }
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }

            sw.Stop();
            ModelEvaluator.DisposeAll();
            
            Console.WriteLine("The file {0} was processed using {1} concurrent model(s) with an error rate of: {2:P2} ({3} error(s) out of {4} record(s)), and a throughput of {5:N2} records/sec", @"Test-28x28_cntk_text.txt", 
                numConcurrentModels, (float)errorCount / count, errorCount, count, (count + errorCount) * 1000.0 / sw.ElapsedMilliseconds);
        }

        /// <summary>
        /// This method shows how to evaluate a trained image classification model, with
        /// explicitly created feature vectors.
        /// </summary>
        public static List<float> EvaluateImageInputUsingFeatureVector()
        {
            List<float> outputs = null;

            try
            {
                // This example requires the RestNet_18 model.
                // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_18.model"/>
                // The model is assumed to be located at: <CNTK>\Examples\Image\Classification\ResNet 
                // along with a sample image file named "zebra.jpg".
                Environment.CurrentDirectory = initialDirectory;

                using (var model = new IEvaluateModelManagedF())
                {
                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", resnetModelFilePath), deviceId: -1);

                    // Prepare input value in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    if (inDims.First().Value != resNetImageSize * resNetImageSize * 3)
                    {
                        throw new CNTKRuntimeException(string.Format("The input dimension for {0} is {1} which is not the expected size of {2}.", inDims.First(), inDims.First().Value, 224 * 224 * 3), string.Empty);
                    }

                    // Transform the image
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(imageFileName));
                    var resized = bmp.Resize(resNetImageSize, resNetImageSize, true);

                    var resizedCHW = resized.ParallelExtractCHW();
                    var inputs = new Dictionary<string, List<float>>() { {inDims.First().Key, resizedCHW } };

                    // We can call the evaluate method and get back the results (single layer output)...
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);
                    outputs = model.Evaluate(inputs, outDims.First().Key);
                }

                // Retrieve the outcome index (so we can compare it with the expected index)
                var max = outputs.Select((value, index) => new { Value = value, Index = index })
                    .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                    .Index;

                Console.WriteLine("EvaluateImageInputUsingFeatureVector: Outcome = {0}", max);
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }
            return outputs;
        }

        /// <summary>
        /// This method shows how to evaluate a trained image classification model, where the 
        /// creation of the CNTK feature vector is happening in native code inside the EvalWrapper.
        /// </summary>
        public static List<float> EvaluateImageInputUsingImageApi()
        {
            List<float> outputs = null;

            try
            {
                // This example requires the RestNet_18 model.
                // The model can be downloaded from <see cref="https://www.cntk.ai/resnet/ResNet_18.model"/>
                // The model is assumed to be located at: <CNTK>\Examples\Image\Classification\ResNet 
                // along with a sample image file named "zebra.jpg".
                Environment.CurrentDirectory = initialDirectory;
                using (var model = new IEvaluateModelManagedF())
                {
                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", resnetModelFilePath), deviceId: -1);

                    // Prepare input value in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.Input);
                    if (inDims.First().Value != resNetImageSize * resNetImageSize * 3)
                    {
                        throw new CNTKRuntimeException(string.Format("The input dimension for {0} is {1} which is not the expected size of {2}.", inDims.First(), inDims.First().Value, 224 * 224 * 3), string.Empty);
                    }

                    // Transform the image
                    Bitmap bmp = new Bitmap(Bitmap.FromFile(imageFileName));
                    var resized = bmp.Resize(resNetImageSize, resNetImageSize, true);
                    // Now evaluate using the alternative API, where we directly pass the
                    // native bitmap data to the unmanaged code.
                    var outDims = model.GetNodeDimensions(NodeGroup.Output);
                    var outputNodeName = outDims.First().Key;
                    outputs = model.EvaluateRgbImage(resized, outputNodeName);
                }

                // Retrieve the outcome index (so we can compare it with the expected index)
                var max = outputs.Select((value, index) => new { Value = value, Index = index })
                    .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                    .Index;

                Console.WriteLine("EvaluateImageInputUsingImageApi: Outcome = {0}", max);
            }
            catch (CNTKException ex)
            {
                OnCNTKException(ex);
            }
            catch (Exception ex)
            {
                OnGeneralException(ex);
            }
            return outputs;
        }

        /// <summary>
        /// Dumps the output to the console
        /// </summary>
        /// <param name="outputs">The structure containing the output layers</param>
        private static void OutputResults(Dictionary<string, List<float>> outputs)
        {
            Console.WriteLine("--- Output results ---");
            foreach (var item in outputs)
            {
                OutputResults(item.Key, item.Value);
            }
        }

        /// <summary>
        /// Dumps the output of a layer to the console
        /// </summary>
        /// <param name="layer">The display name for the layer</param>
        /// <param name="values">The layer values</param>
        private static void OutputResults(string layer, List<float> values)
        {
            if (values == null)
            {
                Console.WriteLine("No Output for layer: {0}", layer);
                return;
            }

            Console.WriteLine("Output layer: {0}", layer);
            foreach (var entry in values)
            {
                Console.WriteLine(entry);
            }
        }

        /// <summary>
        /// Creates a Dictionary for input entries or output allocation 
        /// </summary>
        /// <param name="key">The key for the mapping</param>
        /// <param name="size">The number of element entries associated to the key</param>
        /// <param name="maxValue">The maximum value for random generation values</param>
        /// <returns>A dictionary with a single entry for the key/values</returns>
        static Dictionary<string, List<float>> GetDictionary(string key, int size, int maxValue)
        {
            var dict = new Dictionary<string, List<float>>();
            if (key != string.Empty && size >= 0 && maxValue > 0)
            {
                dict.Add(key, GetFloatArray(size, maxValue));
            }

            return dict;
        }

        /// <summary>
        /// Creats a list of random numbers
        /// </summary>
        /// <param name="size">The size of the list</param>
        /// <param name="maxValue">The maximum value for the generated values</param>
        /// <returns>A list of random numbers</returns>
        static List<float> GetFloatArray(int size, int maxValue)
        {
            List<float> list = new List<float>();
            if (size > 0 && maxValue >= 0)
            {
                Random rnd = new Random();
                list.AddRange(Enumerable.Range(1, size).Select(i => (float)rnd.Next(maxValue)).ToList());
            }

            return list;
        }
    }
}
