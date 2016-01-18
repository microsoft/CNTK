//
// <copyright file="Program.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.MSR.CNTK;

namespace CSEvalClient
{
    class Program
    {
        private static void Main(string[] args)
        {
            Environment.CurrentDirectory = Path.Combine(Environment.CurrentDirectory, @"..\..\Examples\Image\MNIST\Data\");
            Console.WriteLine("Current Directory: {0}", Environment.CurrentDirectory);

            Console.WriteLine("Creating Model Evaluator...");
            var model = new IEvaluateModelManagedF();

            Console.WriteLine("Initializing Model Evaluator...");
            string config = GetConfig();
            model.Init(config);

            Console.WriteLine("Loading Model...");
            string modelFilePath = Path.Combine(Environment.CurrentDirectory, @"..\Output\Models\01_OneHidden");
            model.LoadModel(modelFilePath);

            var inputs = GetDictionary("features", 28 * 28, 255);
            var outputs = GetDictionary("ol.z", 10, 100);

            Console.WriteLine("Press <Enter> to begin evaluating.");
            Console.ReadLine();

            List<float> outputList = null;
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine("Evaluating Model...");
                outputList = model.Evaluate(inputs, "ol.z", 10);  // return results
                model.Evaluate(inputs, outputs);                    // Pass result structure
            }

            Console.WriteLine("Destroying Model...");
            model.Destroy();

            Console.WriteLine("Output contents:");
            foreach (var item in outputs)
            {
                Console.WriteLine(item);
            }

            Console.WriteLine("Press <Enter> to terminate.");
            Console.ReadLine();
        }

        static Dictionary<string, List<float>> GetDictionary(string key, int size, int maxValue)
        {
            return new Dictionary<string, List<float>>() { { key, GetFloatArray(size, maxValue) } };
        }

        static string GetConfig()
        {
            string configFilePath = Path.Combine(Environment.CurrentDirectory,
                    @"..\Config\01_OneHidden.config");

            var lines = System.IO.File.ReadAllLines(configFilePath);
            return string.Join("\n", lines);
        }

        static List<float> GetFloatArray(int size, int maxValue)
        {
            Random rnd = new Random();
            return Enumerable.Range(1, size).Select(i => (float)rnd.Next(maxValue)).ToList();
        }
    }
}
