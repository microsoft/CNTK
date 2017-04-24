//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// MemoryTests.cs -- Memory safety tests
//

using System;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    internal sealed class MemoryTests
    {
        public static DeviceDescriptor Device0;
        public static Axis Axis0;
        public static Variable OutputVar;
        public static Variable OutputVar0;
        public static Variable InputVar0;
        public static Variable ArgumentVar0;
        public static Value OutputVal;

        public static void ValidateObjectReferences(DeviceDescriptor device)
        {
            using (var test = new SetupMemoeryTests())
            {
                test.SetupUsingResetModel(device);
                test.NDArrayViewTest(device);
            }

            Console.WriteLine("\n1. Run: Test saved object references.\n");
            WriteOutputs();

            Console.WriteLine("\n2. Run: Test saved object references.\n");
            WriteOutputs();
        }

        public static void WriteOutputs()
        {
            // enforce GC.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            Console.WriteLine("\nPrint out saved object references.");
            try
            {
                Console.WriteLine("Device0: " + Device0.AsString() + ", Type: " + Device0.Type);
                Console.WriteLine("Axis0: " + Axis0.Name + ", IsStaticAxis: " + Axis0.IsStatic);
                Console.WriteLine("OutputVar: " + OutputVar.AsString() + ", Name: " + OutputVar.Name + ", Kind: " + Utils.VariableKindName(OutputVar.Kind) + ", Shape: " + OutputVar.Shape.AsString());
                Console.WriteLine("OutputVar0: " + OutputVar0.AsString() + ", Name: " + OutputVar0.Name + ", Kind: " + Utils.VariableKindName(OutputVar.Kind) + ", Shape: " + OutputVar0.Shape.AsString());
                Console.WriteLine("InputVar0: " + InputVar0.AsString() + ", Name: " + InputVar0.Name + ", Kind: " + Utils.VariableKindName(OutputVar.Kind) + ", Shape: " + InputVar0.Shape.AsString());
                Console.WriteLine("ArgumentVar0: " + ArgumentVar0.AsString() + ", Name: " + ArgumentVar0.Name + ", Kind: " + Utils.VariableKindName(OutputVar.Kind) + ", Shape: " + ArgumentVar0.Shape.AsString());
                Console.WriteLine("OutputVal: " + ", Device: " + OutputVal.Device.AsString() + ", Storage: " + OutputVal.StorgeFormat + ", Shape: " + OutputVal.Shape.AsString() + "Data:");
                var outputData = OutputVal.GetDenseData<float>(OutputVar);
                CNTKLibraryManagedExamples.PrintOutput(OutputVar.Shape.TotalSize, outputData);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Memory Tests Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
            Console.WriteLine("\nAll saved object references are printed.");
        }
    }
}
