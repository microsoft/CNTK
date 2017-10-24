// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CNTK.V2LibraryCSTests
{
    [TestClass]
    public class FunctionTests
    {
        [TestMethod]
        public void TestSaveAndLoad()
        {
            int channels = 2;
            int imageWidth = 40, imageHeight = 40;
            int[] inputDim = { imageHeight, imageWidth, channels };
            Variable input = CNTKLib.InputVariable(inputDim, DataType.Float, "Images");
            Parameter param = new Parameter(inputDim, DataType.Float, CNTKLib.GlorotUniformInitializer(0.1F, 1, 0), DeviceDescriptor.CPUDevice);
            Function model = CNTKLib.Plus(input, param, "Plus");
            byte[] buffer = model.Save();
            Function loadedModel = Function.Load(buffer, DeviceDescriptor.CPUDevice);
            Assert.AreEqual(loadedModel.Name, model.Name);
            Assert.AreEqual(loadedModel.Inputs.Count, model.Inputs.Count);
            Assert.AreEqual(loadedModel.Inputs[0].Shape, loadedModel.Inputs[0].Shape);
            Assert.AreEqual(loadedModel.Output.Shape, loadedModel.Output.Shape);
        }
    }
}
