//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using System;
using System.Collections.Generic;
using System.IO;
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
            int colIndicesSize = 5;
            var vb = new ValueBuffer<float>(bufferSize);
            Assert.AreEqual(bufferSize, vb.Buffer.Length);
            Assert.IsNull(vb.Indices);
            Assert.IsNull(vb.ColIndices);

            vb = new ValueBuffer<float>(bufferSize, colIndicesSize);
            Assert.AreEqual(bufferSize, vb.Buffer.Length);
            Assert.AreEqual(bufferSize, vb.Indices.Length);
            Assert.AreEqual(colIndicesSize, vb.ColIndices.Length);
        }

        [TestMethod]
        public void EvalManagedVariableSchemaTest()
        {
            VariableSchema sc = new VariableSchema();
            var buffers  = sc.CreateBuffers<float>();
            Assert.AreEqual(0, buffers.Length);

            sc.Add(new VariableLayout(){DataType=DataType.Float32, Name="A", NumElements=5, StorageType = StorageType.Dense});
            buffers = sc.CreateBuffers<float>();
            Assert.AreEqual(5, buffers[0].Buffer.Length);

            sc.Add(new VariableLayout() { DataType = DataType.Float32, Name = "B", NumElements = 10, StorageType = StorageType.Sparse});
            buffers = sc.CreateBuffers<float>();
            Assert.AreEqual(10, buffers[1].Buffer.Length);
            // Although sparse, the Indices and ColIndices are not allocated
            Assert.AreEqual(null, buffers[1].Indices);
            Assert.AreEqual(null, buffers[1].ColIndices);
        }

        [TestMethod]
        public void EvalManagedConstantNetworkTest()
        {
            string modelDefinition = @"precision = ""float""
                traceLevel = 1
                run=NDLNetworkBuilder 
                NDLNetworkBuilder=[ 
                v1 = Constant(1)
                v2 = Constant(2, tag=""output"") 
                ol = Plus(v1, v2, tag=""output"")
                FeatureNodes = (v1)
                ]";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                var outputBuffer = outputSchema.CreateBuffers<float>();
                var inputBuffer = new ValueBuffer<float>[0];

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = { new float[] { 2 }, new float[] {3} };

                Assert.AreEqual(expected.Length, outputBuffer.Length);
                for (int idx = 0; idx < expected.Length; idx++)
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedScalarTimesTest()
        {
            string modelDefinition = @"precision = ""float"" 
                traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[
                i1 = Input(1)
                o1 = Times(Constant(3), i1, tag=""output"") 
                FeatureNodes = (i1)
                ]";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                VariableSchema inputSchema = model.GetInputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                var outputBuffer = outputSchema.CreateBuffers<float>();
                var inputBuffer = inputSchema.CreateBuffers<float>();
                inputBuffer[0].Buffer[0] = 2;
                inputBuffer[0].Size = 1;

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = {new float[]{6}};

                Assert.AreEqual(expected.Length, outputBuffer.Length);
                for (int idx = 0; idx < expected.Length; idx++)
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedSparseTimesTest()
        {
            string modelDefinition = @"deviceId = -1 
                precision = ""float"" traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[ 
                i1 = SparseInput(3)
                o1 = Times(Constant(2, rows=1, cols=3), i1, tag=""output"") 
                FeatureNodes = (i1)
                ]";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                var outputBuffer = new []
                {
                    new ValueBuffer<float>()
                    {
                        Buffer = new float[3],
                        Size = 3
                    }
                };

                var inputBuffer = new []
                {
                    new ValueBuffer<float>()
                    {
                        Buffer = new float[] { 1, 2, 3, 5, 6 },
                        Indices = new [] { 0, 2, 2, 1, 2 },
                        ColIndices = new [] { 0, 2, 2, 5 },
                        Size = 4
                    }
                };

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = { new float[] { 6, 0, 28 } };

                Assert.AreEqual(expected.Length, outputBuffer.Length);
                for (int idx = 0; idx < expected.Length; idx++)
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedScalarTimesDualOutputTest()
        {
            string modelDefinition = @"deviceId = -1 
                precision = ""float""
                traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[
                i1 = Input(1)
                i2 = Input(1)
                o1 = Times(Constant(3), i1, tag=""output"")
                o2 = Times(Constant(5), i1, tag=""output"")
                FeatureNodes = (i1)
                ]";

            using (var model = new ModelEvaluationExtendedF())
            {
                model.CreateNetwork(modelDefinition);

                VariableSchema outputSchema = model.GetOutputSchema();
                VariableSchema inputSchema = model.GetInputSchema();

                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                var outputBuffer = outputSchema.CreateBuffers<float>();
                var inputBuffer = inputSchema.CreateBuffers<float>();
                inputBuffer[0].Buffer[0] = 2;

                // We can call the evaluate method and get back the results...
                model.ForwardPass(inputBuffer, outputBuffer);

                float[][] expected = {new float[]{6}, new float[]{10} };
            
                Assert.AreEqual(expected.Length, outputBuffer.Length);
                for(int idx=0; idx<expected.Length; idx++ )
                {
                    CollectionAssert.AreEqual(expected[idx], outputBuffer[idx].Buffer);
                }
            }
        }

        [TestMethod]
        public void EvalManagedCrossAppDomainExceptionTest()
        {
            var currentPath = Environment.CurrentDirectory;
            var domain = AppDomain.CreateDomain("NewAppDomain");
            var path = Path.Combine(currentPath, "EvalWrapper.dll");
            var t = typeof(CNTKException);
            var instance = (CNTKException)domain.CreateInstanceFromAndUnwrap(path, t.FullName);
            Assert.AreNotEqual(null, instance);
        }
    }
}
