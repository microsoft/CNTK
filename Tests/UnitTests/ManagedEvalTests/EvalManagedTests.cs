//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Drawing;
using Microsoft.VisualStudio.TestTools.UnitTesting;
// Re-use the Resize method defined in the CSEvalClientTest.exe assembly. 
// Strictly speaking, those extensions should live in an assembly of their own.
using Microsoft.MSR.CNTK.Extensibility.Managed.CSEvalClient;
using System.Drawing.Imaging;

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

        private void AssertArgumentException(IEvaluateModelManagedF model, 
            Bitmap image, 
            string outputKey, 
            string expectedParameterName, 
            string expectedMessageText,
            string errorMessage)
        {
            bool exception = false;
            try
            {
                model.EvaluateRgbImage(image, outputKey);
            }
            catch (ArgumentException ex)
            {
                if (ex.ParamName == expectedParameterName && ex.Message.Contains(expectedMessageText))
                {
                    exception = true;
                }
            }
            catch { }
            if (!exception)
            {
                throw new Exception(errorMessage);
            }
        }

        [TestMethod]
        public void EvalManagedImageApiErrorHandling()
        {
            // The width and height of the image that will be fed into the network.
            var expectedSize = 10;
            // Images with correct size and pixel format.
            var correctBmp1 = new Bitmap(expectedSize, expectedSize, PixelFormat.Format24bppRgb);
            var correctBmp2 = new Bitmap(expectedSize, expectedSize, PixelFormat.Format32bppRgb);
            // Image with correct size, but wrong pixel format
            var wrongPixelFormat = new Bitmap(expectedSize, expectedSize, PixelFormat.Format16bppRgb565);
            // Image with wrong size, correct pixel format
            var wrongSize = new Bitmap(expectedSize * 2, expectedSize, PixelFormat.Format24bppRgb);

            var inputVectorSize = expectedSize * expectedSize * 3;
            var modelDefinition = String.Format(@"deviceId = -1 
                precision = ""float""
                traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[
                i1 = Input({0}) # Network must have size expectedSize * expectedSize * 3, for 3 channels
                o1 = Times(Constant(5, rows=1, cols={0}), i1, tag=""output"")
                FeatureNodes = (i1)
                ]", inputVectorSize);
            using (var model = new IEvaluateModelManagedF())
            {
                model.CreateNetwork(modelDefinition);

                model.EvaluateRgbImage(correctBmp1, "o1");
                AssertArgumentException(model, 
                    correctBmp1, 
                    "No such output key", 
                    "outputKey",
                    "not an output node", 
                    "Providing a non-existing output node should fail with an ArgumentException.");
                AssertArgumentException(model,
                    wrongPixelFormat,
                    "o1",
                    "image",
                    "must be one of { Format24bppRgb, Format32bppArgb}",
                    "Images with an unrecognized pixel format should fail with an ArgumentException.");
                AssertArgumentException(model,
                    wrongSize,
                    "o1",
                    "image",
                    "invalid size",
                    "Calling with a wrongly sized image should fail with an ArgumentException.");
            }
        }

    }
}
