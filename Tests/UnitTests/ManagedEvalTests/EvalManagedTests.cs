//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Drawing;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Drawing.Imaging;

namespace Microsoft.MSR.CNTK.Extensibility.Managed.Tests
{
    [TestClass]
    public class EvalManagedTests
    {
        [TestMethod]
        public void EvalManagedEvaluateNoInputTest()
        {
            string modelDefinition = @"precision = ""float""
                traceLevel = 1
                run=NDLNetworkBuilder 
                NDLNetworkBuilder=[ 
                v1 = Constant(1)
                v2 = Constant(2) 
                o1 = Plus(v1, v2, tag=""output"")
                FeatureNodes = (v1)
                ]";

            using (var model = new IEvaluateModelManagedF())
            {
                model.CreateNetwork(modelDefinition);

                var inDims = model.GetNodeDimensions(NodeGroup.Input);
                Assert.AreEqual(inDims.Count(), 0);

                var outDims = model.GetNodeDimensions(NodeGroup.Output);
                Assert.AreEqual(outDims.Count(), 1);
                Assert.AreEqual(outDims.First().Key, "o1");
                Assert.AreEqual(outDims.First().Value, 1);

                var outputVal = model.Evaluate(outDims.First().Key);

                var expected = new List<float>() {3};
                CollectionAssert.AreEqual(expected, outputVal);
            }
        }

        [TestMethod]
        public void EvalManagedEvaluateSingleOutputTest()
        {
            string modelDefinition = @"precision = ""float"" 
                traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[
                i1 = Input(1)
                o1 = Times(Constant(3), i1, tag=""output"") 
                FeatureNodes = (i1)
                ]";

            using (var model = new IEvaluateModelManagedF())
            {
                model.CreateNetwork(modelDefinition);

                var inDims = model.GetNodeDimensions(NodeGroup.Input);
                Assert.AreEqual(inDims.Count(), 1);
                Assert.AreEqual(inDims.First().Key, "i1");
                Assert.AreEqual(inDims.First().Value, 1);

                var inputs = new Dictionary<string, List<float>>()
                    {
                        {inDims.First().Key, new List<float>() {2}}
                    };

                var outDims = model.GetNodeDimensions(NodeGroup.Output);
                Assert.AreEqual(outDims.Count(), 1);
                Assert.AreEqual(outDims.First().Key, "o1");
                Assert.AreEqual(outDims.First().Value, 1);

                var outputs = new Dictionary<string, List<float>>()
                    {
                        {outDims.First().Key, new List<float>() {0}}
                    };

                model.Evaluate(inputs, outputs);

                var expected = new List<float>() {6};
                CollectionAssert.AreEqual(expected, outputs.First().Value);
            }
        }

        [TestMethod]
        public void EvalManagedEvaluateDualOutputsTest()
        {
            string modelDefinition = @"deviceId = -1 
                precision = ""float""
                traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[
                i1 = Input(1)
                o1 = Times(Constant(3), i1, tag=""output"")
                o2 = Times(Constant(5), i1, tag=""output"")
                FeatureNodes = (i1)
                ]";

            using (var model = new IEvaluateModelManagedF())
            {
                model.CreateNetwork(modelDefinition);

                var inDims = model.GetNodeDimensions(NodeGroup.Input);
                Assert.AreEqual(inDims.Count(), 1);
                Assert.AreEqual(inDims.First().Key, "i1");
                Assert.AreEqual(inDims.First().Value, 1);

                var inputs = new Dictionary<string, List<float>>()
                    {
                        {"i1", new List<float>() {2}}
                    };

                var outDims = model.GetNodeDimensions(NodeGroup.Output);
                Assert.AreEqual(outDims.Count(), 2);
                Assert.AreEqual(outDims["o1"], 1);
                Assert.AreEqual(outDims["o2"], 1);

                var outputs = new Dictionary<string, List<float>>()
                    {
                        {"o1", new List<float>() {0}},
                        {"o2", new List<float>() {0}}
                    };

                model.Evaluate(inputs, outputs);

                var expected1 = new List<float>() {6};
                var expected2 = new List<float>() {10};
                CollectionAssert.AreEqual(expected1, outputs["o1"]);
                CollectionAssert.AreEqual(expected2, outputs["o2"]);
            }
        }

        [TestMethod]
        public void EvalManagedEvaluateHiddenLayerTest()
        {
            string modelDefinition = @"deviceId = -1 
                precision = ""float""
                traceLevel = 1
                run=NDLNetworkBuilder
                NDLNetworkBuilder=[
                i1 = Input(1)
                pool5 = Times(Constant(2), i1)
                OutputNodes.z = Plus(pool5, Constant(7), tag=""output"")
                FeatureNodes = (i1)
                ]";

            using (var model = new IEvaluateModelManagedF())
            {
                var desiredOutputLayers = new List<string>() { "pool5", "OutputNodes.z" };

                model.CreateNetwork(modelDefinition, deviceId: -1, outputNodeNames: desiredOutputLayers);

                var inDims = model.GetNodeDimensions(NodeGroup.Input);
                Assert.AreEqual(inDims.Count(), 1);
                Assert.AreEqual(inDims["i1"], 1);

                var inputs = new Dictionary<string, List<float>>()
                    {
                        { "i1", new List<float>() {5} }
                    };

                // We request the output layer names(s) and dimension, we'll use the first one.
                var outDims = model.GetNodeDimensions(NodeGroup.Output);
                Assert.AreEqual(outDims.Count(), 2);
                Assert.AreEqual(outDims["pool5"], 1);
                Assert.AreEqual(outDims["OutputNodes.z"], 1);

                var outputs = new Dictionary<string, List<float>>()
                    {
                        // The order of node name below is different than that returned by outDims,
                        // in order to test whether the output values are correctly mapped to the name.
                        { "pool5", new List<float>() {0} },
                        { "OutputNodes.z", new List<float>() {0} }
                    };

                // We can call the evaluate method and get back the results (single layer)...
                model.Evaluate(inputs, outputs);

                var expected1 = new List<float>() {10};
                var expected2 = new List<float>() {17};
                CollectionAssert.AreEqual(expected1, outputs["pool5"]);
                CollectionAssert.AreEqual(expected2, outputs["OutputNodes.z"]);
            }
        }

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
            Assert.AreEqual(colIndicesSize, vb.Size);
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
        public void EvalManagedUsingSparseValueBufferTest()
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

                var outputDataLength = 3;
                var outputBuffer = new[]
                {
                    new ValueBuffer<float>(outputDataLength)
                };

                var inputData = new float[] { 1, 2, 3, 5, 6 };
                var inputIndices = new [] { 0, 2, 2, 1, 2 };
                var inputColIndices = new [] { 0, 2, 2, 5 };

                var inputBuffer = new[]
                {
                    new ValueBuffer<float>(inputData.Length, inputColIndices.Length)
                };

                inputData.CopyTo(inputBuffer[0].Buffer, 0);
                inputIndices.CopyTo(inputBuffer[0].Indices, 0);
                inputColIndices.CopyTo(inputBuffer[0].ColIndices, 0);

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

            // search for "our" dll, ignoring the version number
            var names = Directory.EnumerateFiles(currentPath, "Cntk.Eval.Wrapper-*.dll");
            var dllpathname = names.FirstOrDefault();

            var domain = AppDomain.CreateDomain("NewAppDomain");
            var t = typeof(CNTKException);
            var instance = (CNTKException)domain.CreateInstanceFromAndUnwrap(dllpathname, t.FullName);
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
                o1 = Times(Constant(1, rows=2, cols={0}), i1, tag=""output"")
                FeatureNodes = (i1)
                ]", inputVectorSize);
            using (var model = new IEvaluateModelManagedF())
            {
                model.CreateNetwork(modelDefinition);

                var output = model.EvaluateRgbImage(correctBmp1, "o1");
                // Network computes 2 simple dot products. 
                Assert.AreEqual(2, output.Count, "Size of output vector");
                // Input image is all zero, output should be too.
                Assert.AreEqual(0.0f, output[0], "OutputVector[0]");
                Assert.AreEqual(0.0f, output[1], "OutputVector[1]");
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

        [TestMethod]
        public void EvalManagedRNNTest()
        {
            string modelDefinition = @"deviceId = -1
                precision = ""float""
                traceLevel = 1 
                run=NDLNetworkBuilder
                NDLNetworkBuilder = [
                LSTMComponent(inputDim, outputDim, cellDim, inputx, cellDimX2, cellDimX3, cellDimX4) = [
                    wx = Parameter(cellDimX4, 0, init = ""uniform"", initValueScale = 1);
                    b = Parameter(cellDimX4, 1, init = ""fixedValue"", value = 0.0);
                    Wh = Parameter(cellDimX4, 0, init = ""uniform"", initValueScale = 1);

                    Wci = Parameter(cellDim, init = ""uniform"", initValueScale = 1);
                    Wcf = Parameter(cellDim, init = ""uniform"", initValueScale = 1);
                    Wco = Parameter(cellDim, init = ""uniform"", initValueScale = 1);

                    dh = PastValue(outputDim, output, timeStep = 1);
                    dc = PastValue(cellDim, ct, timeStep = 1);

                    wxx = Times(wx, inputx);
                    wxxpb = Plus(wxx, b);

                    whh = Times(wh, dh);

                    wxxpbpwhh = Plus(wxxpb, whh)

                    G1 = RowSlice(0, cellDim, wxxpbpwhh)
                    G2 = RowSlice(cellDim, cellDim, wxxpbpwhh)
                    G3 = RowSlice(cellDimX2, cellDim, wxxpbpwhh);
                    G4 = RowSlice(cellDimX3, cellDim, wxxpbpwhh);

                    Wcidc = DiagTimes(Wci, dc);
                    it = Sigmoid(Plus(G1, Wcidc));

                    bit = ElementTimes(it, Tanh(G2));

                    Wcfdc = DiagTimes(Wcf, dc);
                    ft = Sigmoid(Plus(G3, Wcfdc));

                    bft = ElementTimes(ft, dc);

                    ct = Plus(bft, bit);

                    Wcoct = DiagTimes(Wco, ct);
                    ot = Sigmoid(Plus(G4, Wcoct));

                    mt = ElementTimes(ot, Tanh(ct));

                    Wmr = Parameter(outputDim, cellDim, init = ""uniform"", initValueScale = 1);
                    output = Times(Wmr, mt);
                ]

                i1 = Input(4)
                    o1 = LSTMComponent(4, 4, 1, i1, 2, 3, 4)
                    FeatureNodes = (i1)
                    outputNodes = (o1)
                ]";

            using (var model = new ModelEvaluationExtendedF())
            {
                int featDim = 4;
                int labelDim = 4;

                model.CreateNetwork(modelDefinition);

                VariableSchema inputSchema = model.GetInputSchema();
                VariableSchema outputSchema = model.GetOutputSchema();
                model.StartForwardEvaluation(outputSchema.Select(s => s.Name).ToList<string>());

                // Allocate the output values layer
                var outputBuffer = outputSchema.CreateBuffers<float>();
                var inputBuffer = inputSchema.CreateBuffers<float>();

                for (var i = 0; i < featDim; i++)
                    inputBuffer[0].Buffer[i] = (float)i;

                int scaler = 100000;
                var result = new int[labelDim];
                int[] expected = { 50, 10, 54, 55 };

                // the first pass with reset
                model.ForwardPass(inputBuffer, outputBuffer);

                for (var i = 0; i < labelDim; i++)
                    result[i] = (int)(outputBuffer[0].Buffer[i] * scaler);
                CollectionAssert.AreEqual(expected, result);

                // the second pass with reset
                model.ForwardPass(inputBuffer, outputBuffer);

                for (var i = 0; i < labelDim; i++)
                    result[i] = (int)(outputBuffer[0].Buffer[i] * scaler);
                CollectionAssert.AreEqual(expected, result);

                // another pass with reset
                model.ForwardPass(inputBuffer, outputBuffer, true);

                for (var i = 0; i < labelDim; i++)
                    result[i] = (int)(outputBuffer[0].Buffer[i] * scaler);
                CollectionAssert.AreEqual(expected, result);

                // pass w/o reset
                model.ForwardPass(inputBuffer, outputBuffer, false);
                for (var i = 0; i < labelDim; i++)
                    result[i] = (int)(outputBuffer[0].Buffer[i] * scaler);

                expected = new int[] { 13, 2, 14, 14 };
                CollectionAssert.AreEqual(expected, result);

                // another pass w/o reset
                model.ForwardPass(inputBuffer, outputBuffer, false);
                for (var i = 0; i < labelDim; i++)
                    result[i] = (int)(outputBuffer[0].Buffer[i] * scaler);

                expected = new int[] { -4, 0, -4, -4 };
                CollectionAssert.AreEqual(expected, result);
            }
        }
    }
}
