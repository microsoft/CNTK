//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ModelEvaluator.cs -- wrapper for a network so it can be evaluated one call at a time.
// 
// THIS CODE IS FOR ILLUSTRATION PURPOSES ONLY. NOT FOR PRODUCTION.
//

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient
{
    /// <summary>
    /// This class provides an Eval model wrapper to restrict model evaluation calls to one at a time.
    /// </summary>
    /// <remarks>
    /// This class is not thread-safe except through the static methods.
    /// Each ModelEvaluator instance wraps an Eval model, and exposes the Evaluate method for either
    /// a vector of inputs or a record string.
    /// The static interface provides the management of the concurrency of the models and restricts
    /// the evaluations to a single thread.
    /// </remarks>
    public sealed class ModelEvaluator
    {
        /// <summary>
        /// The cntk model evaluation instance
        /// </summary>
        private readonly IEvaluateModelManagedF m_model;

        /// <summary>
        /// The input layer key
        /// </summary>
        private readonly string m_inKey;

        /// <summary>
        /// The output layer key
        /// </summary>
        private readonly string m_outKey;

        /// <summary>
        /// The model instance number
        /// </summary>
        private readonly int m_modelInstance;

        /// <summary>
        /// The input buffer
        /// </summary>
        private Dictionary<string, List<float>> m_inputs;

        /// <summary>
        /// Indicates if the object is diposed
        /// </summary>
        private static bool Disposed
        {
            get;
            set;
        }

        /// <summary>
        /// The ModelEvaluator's models to manage
        /// </summary>
        private static readonly BlockingCollection<ModelEvaluator> Models = new BlockingCollection<ModelEvaluator>();

        /// <summary>
        /// Initializes the Model Evaluator to process multiple models concurrently
        /// </summary>
        /// <param name="numConcurrentModels">The number of concurrent models</param>
        /// <param name="modelFilePath">The model file path to load the model from</param>
        /// <param name="numThreads"></param>
        public static void Initialize(int numConcurrentModels, string modelFilePath, int numThreads = 1)
        {
            if (Disposed)
            {
                throw new CNTKRuntimeException("Model Evaluator has been disposed", string.Empty);
            }

            for (int i = 0; i < numConcurrentModels; i++)
            {
                Models.Add(new ModelEvaluator(modelFilePath, numThreads, i));
            }
            
            Disposed = false;
        }

        /// <summary>
        /// Disposes of all models
        /// </summary>
        public static void DisposeAll()
        {
            Disposed = true;

            foreach (var model in Models)
            {
                model.Dispose();
            }
            
            Models.Dispose();
        }

        /// <summary>
        /// Evaluates a record containing the input data and the expected outcome value
        /// </summary>
        /// <param name="record">A tab-delimited string with the first entry being the expected value.</param>
        /// <returns>true if the outcome is as expected, false otherwise</returns>
        public static bool Evaluate(string record)
        {
            var model = Models.Take();
            try
            {
                var outcome = model.EvaluateRecord(record);
                return outcome;
            }
            finally
            { 
                Models.Add(model); 
            }
        }

        /// <summary>
        /// Evaluated a vector and returns the output vector
        /// </summary>
        /// <param name="inputs">The input vector</param>
        /// <returns>The output vector</returns>
        public static List<float> Evaluate(List<float> inputs)
        {
            var model = Models.Take();
            try
            {
                var outcome = model.EvaluateInput(inputs);
                return outcome;
            }
            finally
            {
                Models.Add(model);
            }
        }

        /// <summary>
        /// Creates an instance of the <see cref="ModelEvaluator"/> class.
        /// </summary>
        /// <param name="modelFilePath">The model file path</param>
        /// <param name="numThreads">The number of concurrent threads for the model</param>
        /// <param name="id">A unique id for the model</param>
        /// <remarks>The id is used only for debugging purposes</remarks>
        private ModelEvaluator(string modelFilePath, int numThreads, int id)
        {
            m_modelInstance = id;

            m_model = new IEvaluateModelManagedF();

            // Configure the model to run with a specific number of threads
            m_model.Init(string.Format("numCPUThreads={0}", numThreads));

            // Load model
            m_model.CreateNetwork(string.Format("modelPath=\"{0}\"", modelFilePath), deviceId: -1);

            // Generate random input values in the appropriate structure and size
            var inDims = m_model.GetNodeDimensions(NodeGroup.Input);
            m_inKey = inDims.First().Key;
            m_inputs = new Dictionary<string, List<float>>() { { m_inKey, null } };

            // We request the output layer names(s) and dimension, we'll use the first one.
            var outDims = m_model.GetNodeDimensions(NodeGroup.Output);
            m_outKey = outDims.First().Key;
        }

        /// <summary>
        /// Evaluates a test record
        /// </summary>
        /// <param name="record">A tab-delimited string containing as the first entry the expected outcome, values after that are the input data</param>
        /// <returns>true if the record's expected outcome value matches the computed value</returns>
        private bool EvaluateRecord(string record)
        {
            // The first value in the line is the expected label index for the record's outcome
            int expected = int.Parse(record.Substring(0, record.IndexOf('\t')));
            m_inputs[m_inKey] =
                record.Substring(record.IndexOf('\t') + 1).Split('\t').Select(float.Parse).ToList();

            // We can call the evaluate method and get back the results (single layer)...
            var outputs = m_model.Evaluate(m_inputs, m_outKey);

            // Retrieve the outcome index (so we can compare it with the expected index)
            int index = 0;
            var max = outputs.Select(v => new { Value = v, Index = index++ })
                .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
                .Index;

            return (expected == max);
        }

        /// <summary>
        /// Evaluates an input vector against the model as the first defined input layer, and returns the first defined output layer
        /// </summary>
        /// <param name="inputs">Input vector</param>
        /// <returns>The output vector</returns>
        private List<float> EvaluateInput(List<float> inputs)
        {
            return m_model.Evaluate(new Dictionary<string, List<float>>() { { m_inKey, inputs } }, m_outKey);
        }

        /// <summary>
        /// Disposes of the resources
        /// </summary>
        private void Dispose()
        {
            m_model.Dispose();
        }
    }
}
