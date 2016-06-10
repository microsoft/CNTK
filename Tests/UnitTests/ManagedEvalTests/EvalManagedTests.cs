//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.MSR.CNTK.Extensibility.Managed;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.MSR.CNTK.Managed;

namespace Microsoft.MSR.CNTK.Managed.Tests
{
    [TestClass]
    public class EvalManagedTests
    {
        [TestMethod]
        public void EvalManagedValuesBufferTest()
        {
            int size = 2;
            var vb = new ValueBuffer<float>(size);
            Assert.AreEqual(size, vb.m_buffer.Length);
            Assert.AreEqual(size, vb.m_indices.Length);
            Assert.AreEqual(size, vb.m_colIndices.Length);
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

            using (var model = new IEvaluateModelExtendedManagedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.m_name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = outputSchema.CreateBuffers<float>();
                List<ValueBuffer<float>> inputBuffer = new List<ValueBuffer<float>>();

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                List<float> expected = new List<float>() { 2, 3 /* 1 + 2 */ };
                var buf = outputBuffer[0].m_buffer;

                Assert.AreEqual(string.Join(" - ", expected), string.Join(" - ", outputBuffer.Select(b => string.Join(", ", b.m_buffer)).ToList<string>()));
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

            using (var model = new IEvaluateModelExtendedManagedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                VariableSchema inputSchema = model.GetInputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.m_name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = outputSchema.CreateBuffers<float>();
                List<ValueBuffer<float>> inputBuffer = inputSchema.CreateBuffers<float>();
                inputBuffer[0].m_buffer = new float[]{ 2 };

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                List<float> expected = new List<float>() { 6 };
                var buf = outputBuffer[0].m_buffer;

                Assert.AreEqual(string.Join(" - ", expected), string.Join(" - ", outputBuffer.Select(b => string.Join(", ", b.m_buffer)).ToList<string>()));
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

            using (var model = new IEvaluateModelExtendedManagedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                VariableSchema inputSchema = model.GetInputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.m_name).ToList<string>());

                List<ValueBuffer<float>> outputBuffer = outputSchema.CreateBuffers<float>();
                List<ValueBuffer<float>> inputBuffer = inputSchema.CreateBuffers<float>();
                inputBuffer[0].m_buffer = new float[] {2};

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = {new float[]{6}, new float[]{10} };
            
                Assert.AreEqual(expected.Length, outputBuffer.Count);
                for(int idx=0; idx<expected.Length; idx++ )
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].m_buffer);
                }
            }
        }
    }
}
