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
    /// Program for running model evaluations using the CLIWrapper
    /// </summary>
    /// <description>
    /// This program is a managed client using the CLIWrapper to run the model evaluator in CTNK.
    /// It uses one of the examples provided in CNTK for evaluating the model associated with the example.
    /// In order to run this program the model must already exist in the example. To create the model,
    /// first run the example in <CNTK>/Examples/Image/MNIST. Once the model file 01_OneHidden is created,
    /// you can run this client.
    /// This client shows two methods for obtaining the output results from the evaluation, the first as
    /// return values from the Evaluate method call (which only returns a single layer output), and the second
    /// by passing the allocated output layers to the evaluate method.
    /// </description>
    class Program
    {
        /// <summary>
        /// Program entry point
        /// </summary>
        /// <param name="args">Program arguments (ignored)</param>
        private static void Main(string[] args)
        {
            try
            {
                // The examples assume the executable is running from the data folder
                // We switch the current directory to the data folder (assuming the executable is in the <CNTK>/x64/Debug|Release folder
                Environment.CurrentDirectory = Path.Combine(Environment.CurrentDirectory, @"..\..\Examples\Image\MNIST\Data\");
                
                Dictionary<string, List<float>> outputs;

                using (var model = new IEvaluateModelManagedF())
                {
                    // Initialize model evaluator
                    string config = GetConfig();
                    model.Init(config);

                    // Load model
                    string modelFilePath = Path.Combine(Environment.CurrentDirectory, @"..\Output\Models\01_OneHidden");
                    model.LoadModel(modelFilePath);

                    // Generate random input values in the appropriate structure and size
                    var inputs = GetDictionary("features", 28*28, 255);
                    
                    // We can call the evaluate method and get back the results (single layer)...
                    // List<float> outputList = model.Evaluate(inputs, "ol.z", 10);

                    // ... or we can preallocate the structure and pass it in (multiple output layers)
                    outputs = GetDictionary("ol.z", 10, 1);
                    model.Evaluate(inputs, outputs);                    
                }
                
                Console.WriteLine("--- Output results ---");
                foreach (var item in outputs)
                {
                    Console.WriteLine("Output layer: {0}", item.Key);
                    foreach (var entry in item.Value)
                    {
                        Console.WriteLine(entry);
                    }
                }
            }
            catch (CNTKException ex)
            {
                Console.WriteLine("Error: {0}\nNative CallStack: {1}\n Inner Exception: {2}", ex.Message, ex.NativeCallStack, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
            }

            Console.WriteLine("Press <Enter> to terminate.");
            Console.ReadLine();
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
        /// Reads the configuration file and returns the contents as a string
        /// </summary>
        /// <returns>The content of the configuration file</returns>
        static string GetConfig()
        {
            string configFilePath = Path.Combine(Environment.CurrentDirectory,
                    @"..\Config\01_OneHidden.cntk");

            var lines = System.IO.File.ReadAllLines(configFilePath);
            return string.Join("\n", lines);
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
