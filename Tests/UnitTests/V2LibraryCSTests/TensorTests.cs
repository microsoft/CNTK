// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CNTK.V2LibraryCSTests
{
    [TestClass]
    public class TensorTests
    {
        [TestMethod]
        public void TestShapeSpecialDimensions()
        {
            NDShape shapeWithInferredDimension = NDShape.CreateNDShape(new int[] { 3, 7, NDShape.InferredDimension });
            Assert.IsTrue(shapeWithInferredDimension.HasInferredDimension);
            Assert.AreEqual(shapeWithInferredDimension[2], NDShape.InferredDimension);

            NDShape shapeWithFreeDimension = NDShape.CreateNDShape(new int[] { 3, 7, NDShape.FreeDimension });
            Assert.IsTrue(shapeWithFreeDimension.HasFreeDimension);
            Assert.AreEqual(shapeWithFreeDimension[2], NDShape.FreeDimension);
        }
    }
}
