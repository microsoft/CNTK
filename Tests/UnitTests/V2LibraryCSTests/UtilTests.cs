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
            Assert.AreEqual(Utils.GetMaxNumCPUThreads(), 2, 
                "Utils.SetMaxNumCPUThreads(2) should set max number of threads to 2.");

            Utils.SetMaxNumCPUThreads(maxThreads);
            Assert.AreEqual(Utils.GetMaxNumCPUThreads(), maxThreads, 
                $"Utils.SetMaxNumCPUThreads({maxThreads}) should reset max number of threads to {maxThreads}.");

            var level = Utils.GetTraceLevel();
            Utils.SetTraceLevel(TraceLevel.Info);
            Assert.AreEqual(Utils.GetTraceLevel(), TraceLevel.Info, 
                "Utils.SetTraceLevel(TraceLevel.Info) should set TraceLevel to TraceLevel.Info.");
            Console.WriteLine("TraceLevel: before: " + level + ", after " + Utils.GetTraceLevel());
            Utils.SetTraceLevel(level);
            Assert.AreEqual(Utils.GetTraceLevel(), level,
                $"Utils.SetTraceLevel({level}) should reset TraceLevel to {level}.");

            Assert.AreEqual(Utils.DataTypeName(DataType.Float), "Float");
            Assert.AreEqual(Utils.DataTypeSize(DataType.Double), 8);
            Assert.AreEqual(Utils.DeviceKindName(DeviceDescriptor.CPUDevice.Type), "CPU");
            Assert.AreEqual(Utils.DeviceKindName(DeviceKind.GPU), "GPU");
            Assert.IsFalse(Utils.IsSparseStorageFormat(StorageFormat.Dense));
            Assert.IsTrue(Utils.IsSparseStorageFormat(StorageFormat.SparseCSC));
            Assert.IsTrue(Utils.IsSparseStorageFormat(StorageFormat.SparseBlockCol));
            Assert.AreEqual(Utils.VariableKindName(VariableKind.Constant), "Constant");
            Assert.AreEqual(Utils.VariableKindName(VariableKind.Placeholder), "Placeholder");
            Assert.AreEqual(Utils.VariableKindName(VariableKind.Input), "Input");
            Assert.AreEqual(Utils.VariableKindName(VariableKind.Output), "Output");
            Assert.AreEqual(Utils.VariableKindName(VariableKind.Parameter), "Parameter");
        }
    }
}
