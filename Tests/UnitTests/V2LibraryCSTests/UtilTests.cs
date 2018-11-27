// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CNTK.V2LibraryCSTests
{
    [TestClass]
    public class UtilTests
    {
        [TestMethod]
        public void TestUtils()
        {
            int maxThreads = Utils.GetMaxNumCPUThreads();
            Utils.SetMaxNumCPUThreads(2);
            Assert.AreEqual(2, Utils.GetMaxNumCPUThreads(), 
                "Utils.SetMaxNumCPUThreads(2) should set max number of threads to 2.");

            Utils.SetMaxNumCPUThreads(maxThreads);
            Assert.AreEqual(maxThreads, Utils.GetMaxNumCPUThreads(),
                $"Utils.SetMaxNumCPUThreads({maxThreads}) should reset max number of threads to {maxThreads}.");

            var level = Utils.GetTraceLevel();
            Utils.SetTraceLevel(TraceLevel.Info);
            Assert.AreEqual(TraceLevel.Info, Utils.GetTraceLevel(),
                "Utils.SetTraceLevel(TraceLevel.Info) should set TraceLevel to TraceLevel.Info.");
            Console.WriteLine("TraceLevel: before: " + level + ", after " + Utils.GetTraceLevel());
            Utils.SetTraceLevel(level);
            Assert.AreEqual(level, Utils.GetTraceLevel(),
                $"Utils.SetTraceLevel({level}) should reset TraceLevel to {level}.");

            Assert.AreEqual("Float", Utils.DataTypeName(DataType.Float));
            Assert.AreEqual(8, Utils.DataTypeSize(DataType.Double));
            Assert.AreEqual("CPU", Utils.DeviceKindName(DeviceDescriptor.CPUDevice.Type));
            Assert.AreEqual("GPU", Utils.DeviceKindName(DeviceKind.GPU));
            Assert.IsFalse(Utils.IsSparseStorageFormat(StorageFormat.Dense));
            Assert.IsTrue(Utils.IsSparseStorageFormat(StorageFormat.SparseCSC));
            Assert.IsTrue(Utils.IsSparseStorageFormat(StorageFormat.SparseBlockCol));
            Assert.AreEqual("Constant", Utils.VariableKindName(VariableKind.Constant));
            Assert.AreEqual("Placeholder", Utils.VariableKindName(VariableKind.Placeholder));
            Assert.AreEqual("Input", Utils.VariableKindName(VariableKind.Input));
            Assert.AreEqual("Output", Utils.VariableKindName(VariableKind.Output));
            Assert.AreEqual("Parameter", Utils.VariableKindName(VariableKind.Parameter));
        }
    }
}
