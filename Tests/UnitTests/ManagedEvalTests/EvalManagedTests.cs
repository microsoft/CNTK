//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Microsoft.MSR.CNTK.Extensibility.Managed.Tests
{
    [TestClass]
    public class EvalManagedTests
    {
        [TestMethod]
        public void EvalManagedValuesBufferTest()
        {
            int bufferSize = 2;
            int indicesSize = 3;
            int colIndicesSize = 5;
            var vb = new ValueBuffer<float>(bufferSize);
            Assert.AreEqual(bufferSize, vb.Buffer.Length);
            Assert.IsNull(vb.Indices);
            Assert.IsNull(vb.ColIndices);

            vb = new ValueBuffer<float>(bufferSize, indicesSize, colIndicesSize);
            Assert.AreEqual(bufferSize, vb.Buffer.Length);
            Assert.AreEqual(indicesSize, vb.Indices.Length);
            Assert.AreEqual(colIndicesSize, vb.ColIndices.Length);
        }

        [TestMethod]
        public void EvalManagedConstantNetworkTest()
        {
            string modelDefinition = "precision = \"float\" \n" +
                "traceLevel = 1 \n" +
                "run=NDLNetworkBuilder \n" +
                "NDLNetworkBuilder=[ \n" +
                "v1 = Constant(1) \n" +
                "v2 = Constant(2, tag=\"output\") \n" +
                "ol = Plus(v1, v2, tag=\"output\") \n" +
                "FeatureNodes = (v1) \n" +
                "] \n";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = outputSchema.CreateBuffers<float>();
                List<ValueBuffer<float>> inputBuffer = new List<ValueBuffer<float>>();

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = { new float[] { 2 }, new float[] {3} };

                Assert.AreEqual(expected.Length, outputBuffer.Count);
                for (int idx = 0; idx < expected.Length; idx++)
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedScalarTimesTest()
        {
            string modelDefinition = "precision = \"float\" \n" +
                "traceLevel = 1 \n" +
                "run=NDLNetworkBuilder \n" +
                "NDLNetworkBuilder=[ \n" +
                "i1 = Input(1) \n" +
                "o1 = Times(Constant(3), i1, tag=\"output\") \n" +
                "FeatureNodes = (i1) \n" +
                "] \n";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                VariableSchema inputSchema = model.GetInputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = outputSchema.CreateBuffers<float>();
                List<ValueBuffer<float>> inputBuffer = inputSchema.CreateBuffers<float>();
                inputBuffer[0].Buffer[0] = 2;

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = {new float[]{6}};

                Assert.AreEqual(expected.Length, outputBuffer.Count);
                for (int idx = 0; idx < expected.Length; idx++)
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedSparseTimesTest()
        {
            string modelDefinition = "deviceId = -1 \n" + 
                "precision = \"float\" \n" +
                "traceLevel = 1 \n" +
                "run=NDLNetworkBuilder \n" +
                "NDLNetworkBuilder=[ \n" +
                "i1 = SparseInput(3) \n" +
                "o1 = Times(Constant(2, rows=1, cols=3), i1, tag=\"output\") \n" +
                "FeatureNodes = (i1) \n" +
                "] \n";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = new List<ValueBuffer<float>>()
                {
                    new ValueBuffer<float>()
                    {
                        Buffer = new float[3]
                    }
                };

                List<ValueBuffer<float>> inputBuffer = new List<ValueBuffer<float>>()
                {
                    new ValueBuffer<float>()
                    {
                        Buffer = new float[] { 1, 2, 3, 5, 6 },
                        Indices = new [] { 0, 2, 2, 1, 2 },
                        ColIndices = new [] { 0, 2, 2, 5 }
                    }
                };

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = { new float[] { 6, 0, 28 } };

                Assert.AreEqual(expected.Length, outputBuffer.Count);
                for (int idx = 0; idx < expected.Length; idx++)
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedScalarTimesDualOutputTest()
        {
            string modelDefinition = "deviceId = -1 \n" +
                "precision = \"float\" \n" +
                "traceLevel = 1 \n" +
                "run=NDLNetworkBuilder \n" +
                "NDLNetworkBuilder=[ \n" +
                "i1 = Input(1) \n" +
                "i2 = Input(1) \n" +
                "o1 = Times(Constant(3), i1, tag=\"output\") \n" +
                "o2 = Times(Constant(5), i1, tag=\"output\") \n" +
                "FeatureNodes = (i1) \n" +
                "] \n";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                VariableSchema inputSchema = model.GetInputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = outputSchema.CreateBuffers<float>();
                List<ValueBuffer<float>> inputBuffer = inputSchema.CreateBuffers<float>();
                inputBuffer[0].Buffer[0] = 2;

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = {new float[]{6}, new float[]{10} };
            
                Assert.AreEqual(expected.Length, outputBuffer.Count);
                for(int idx=0; idx<expected.Length; idx++ )
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }
    }
}
