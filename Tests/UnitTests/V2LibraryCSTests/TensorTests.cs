using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CNTK;

namespace V2LibraryCSTests
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
