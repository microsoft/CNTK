//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs -- main C# file that contains client code to call the CLI Wrapper class.
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient
{
    /// <summary>
    /// Program for demonstrating how to run model evaluations using the CLIWrapper
    /// </summary>
    /// <description>
    /// This program is a managed client using the CLIWrapper to run the model evaluator in CTNK.
    /// There are four cases shown in this program related to model loading/network creation and evaluation.
    /// The first two use the trained model from one of the examples provided in the CNTK source code.
    /// In order to run this program the model must already exist in the example. To create the model,
    /// first run the example in <CNTK>/Examples/Image/MNIST. Once the model file 01_OneHidden is created,
    /// you can run this client.
    /// The last two cases show how to evaluate a network without first training the model. This is accomplished
    /// by building the network and evaluating a single forward pass.
    /// This program also shows how to obtaining the output results from the evaluation, either as the default output layer,
    /// or by specifying one or more layers as outputs.
    /// </description>
    class Program
    {
        private static string initialDirectory;

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
            
            Console.WriteLine("\n====== EvaluateNetworkSingleLayer ========");
            EvaluateNetworkSingleLayer();

            Console.WriteLine("\n====== EvaluateNetworkSingleLayerNoInput ========");
            EvaluateNetworkSingleLayerNoInput();

            Console.WriteLine("Press <Enter> to terminate.");
            Console.ReadLine();
        }

        /// <summary>
        /// Evaluates a trained model and obtains a single layer output
        /// </summary>
        private static void EvaluateModelSingleLayer()
        {
            try
            {
                string outputLayerName;

                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                Environment.CurrentDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Image\MNIST\Data\");
                List<float> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Load model
                    string modelFilePath = Path.Combine(Environment.CurrentDirectory, @"..\Output\Models\01_OneHidden");
                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelFilePath), deviceId:-1);

                    // Generate random input values in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.nodeInput);
                    var inputs = GetDictionary(inDims.First().Key, inDims.First().Value, 255);
                    
                    // We request the output layer names(s) and dimension, we'll use the first one.
                    var outDims = model.GetNodeDimensions(NodeGroup.nodeOutput);
                    outputLayerName = outDims.First().Key;
                    // We can call the evaluate method and get back the results (single layer)...
                    outputs = model.Evaluate(inputs, outputLayerName);
                }

                OutputResults(outputLayerName, outputs);
            }
            catch (CNTKException ex)
            {
                Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
        }

        /// <summary>
        /// Evaluates a trained model and obtains multiple layers output
        /// </summary>
        private static void EvaluateModelMultipleLayers()
        {
            try
            {
                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                Environment.CurrentDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Image\MNIST\Data\");

                Dictionary<string, List<float>> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Load model
                    string modelFilePath = Path.Combine(Environment.CurrentDirectory, @"..\Output\Models\01_OneHidden");
                    model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelFilePath), deviceId:-1);

                    // Generate random input values in the appropriate structure and size
                    var inDims = model.GetNodeDimensions(NodeGroup.nodeInput);
                    var inputs = GetDictionary(inDims.First().Key, inDims.First().Value, 255);

                    // We request the output layer names(s) and dimension, we'll use the first one.
                    var outDims = model.GetNodeDimensions(NodeGroup.nodeOutput);
                    string outputLayerName = outDims.First().Key;

                    // We can preallocate the output structure and pass it in (multiple output layers)
                    outputs = GetDictionary(outputLayerName, outDims[outputLayerName], 1);
                    model.Evaluate(inputs, outputs);
                }

                OutputResults(outputs);
            }
            catch (CNTKException ex)
            {
                Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
        }

        /// <summary>
        /// Evaluates a network (without a model) and obtains a single layer output
        /// </summary>
        private static void EvaluateNetworkSingleLayer()
        {
            try
            {
                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                string workingDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Other\Simple2d\Config");
                Environment.CurrentDirectory = initialDirectory;

                List<float> outputs;
                string outputLayerName;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Create the network
                    // This network (AddOperatorConstant.cntk) is a simple network consisting of a single binary operator (Plus)
                    // operating over a single input and a constant
                    string networkDescription = File.ReadAllText(Path.Combine(workingDirectory, @"AddOperatorConstant.cntk"));
                    model.CreateNetwork(networkDescription, deviceId:-1);

                    // Generate random input value in the appropriate structure and size
                    var inputs = new Dictionary<string, List<float>>() { { "features", new List<float>() { 1.0f } } };

                    // We can call the evaluate method and get back the results (single layer)...
                    var outDims = model.GetNodeDimensions(NodeGroup.nodeOutput);
                    outputLayerName = outDims.First().Key;
                    outputs = model.Evaluate(inputs, outputLayerName);
                }

                OutputResults(outputLayerName, outputs);
            }
            catch (CNTKException ex)
            {
                Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
        }

        /// <summary>
        /// Evaluates a network (without a model and without input) and obtains a single layer output
        /// </summary>
        private static void EvaluateNetworkSingleLayerNoInput()
        {
            try
            {
                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                string workingDirectory = Path.Combine(initialDirectory, @"..\..\Examples\Other\Simple2d\Config");
                Environment.CurrentDirectory = initialDirectory;

                List<float> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Create the network
                    // This network (AddOperatorConstantNoInput.cntk) is a simple network consisting of a single binary operator (Plus)
                    // operating over a two constants, therefore no input is necessary.
                    string networkDescription = File.ReadAllText(Path.Combine(workingDirectory, @"AddOperatorConstantNoInput.cntk"));
                    model.CreateNetwork(networkDescription, deviceId:-1);

                    // We can call the evaluate method and get back the results (single layer)...
                    outputs = model.Evaluate("ol", 1);
                }

                OutputResults("ol", outputs);
            }
            catch (CNTKException ex)
            {
                Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }
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
