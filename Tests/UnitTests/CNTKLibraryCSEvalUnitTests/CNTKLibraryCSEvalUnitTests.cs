using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CNTK;

namespace CNTKLibraryCSEvalUnitTests
{
    [TestClass]
    public class CNTKLibraryCSEvalUnitTests
    {
        private static DeviceDescriptor device;
        private static Function model1;
        private static Function model1Clone;
        private static Function model2;
        private static string modelFilePath = "resnet20.dnn";

        [ClassInitialize()]
        public static void ClassInit(TestContext context)
        {
            // Todo: use a test model which has known properties, e.g inputs, outputs.
            device = ShouldRunOnGpu() ? DeviceDescriptor.GPUDevice(0) : DeviceDescriptor.CPUDevice;

            model1 = Function.LoadModel(modelFilePath, device);
            model1Clone = model1.Clone();
            model2 = Function.LoadModel(modelFilePath, device);
        }

        [TestMethod]
        public void VariableTests()
        {
            var var1 = model1.Output;
            var var1Clone = model1Clone.Output;
            var var2 = model2.Output;

            // Test equality of variable
            var newVar1 = var1;
            Assert.AreEqual(var1, newVar1);
            Assert.AreNotEqual(var1, var1Clone);
            Assert.AreNotEqual(var1, var2);

            // Test varaible properties


        }

        public static bool ShouldRunOnGpu()
        {
#if CPUONLY
            return false;
#else
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "gpu"));
#endif
        }

        static bool ShouldRunOnCpu()
        {
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "cpu"));
        }
    }
}
