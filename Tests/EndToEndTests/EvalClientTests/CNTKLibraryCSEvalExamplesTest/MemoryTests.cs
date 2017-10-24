//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// MemoryTests.cs -- Memory safety tests
//

using System;
using CNTK;
using System.Collections.Generic;
using System.Linq;

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
                Console.WriteLine("OutputVal: " + ", Device: " + OutputVal.Device.AsString() + ", Storage: " + OutputVal.StorageFormat + ", Shape: " + OutputVal.Shape.AsString() + "Data:");
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

        public static void ValueCopyToSparseCSCTest<T>(DeviceDescriptor device)
        {
            try
            {
                // uint numAxes = 2;
                int dimension = 3;
                IEnumerable<int> dims = new List<int>() { dimension };
                var sampleShape = NDShape.CreateNDShape(dims);
                int seqLen = 1;

                int[] colStarts = { 0, 1 };
                int[] rowIndices = { 2 };
                T[] nonZeroValues = { (typeof(T) == typeof(float)) ? (T)(object)(0.5F) : (T)(object)(0.5) };


                bool sequenceStartFlag = true;
                bool readOnly = false;
                var sparseValue = Value.CreateSequence<T>(sampleShape, seqLen, colStarts, rowIndices, nonZeroValues, sequenceStartFlag, device, readOnly);

                NDArrayView value = null;
                bool needsGradient = false;
                var dynamicAxes = new AxisVector(0);
                bool isSparse = true;
                var sampleVariable = new Variable(sampleShape, VariableKind.Output,
                    (typeof(T) == typeof(float)) ? DataType.Float : DataType.Double, value, needsGradient,
                    dynamicAxes, isSparse, "sampleVariable", "sampleVariableUid");

                int output_seqLen;
                IList<int> output_colsStarts;
                IList<int> output_rowIndices;
                IList<T> output_nonZeroValues;
                int output_numNonZeroValues;
                sparseValue.GetSparseData<T>(sampleVariable, out output_seqLen,
                    out output_colsStarts, out output_rowIndices, out output_nonZeroValues, out output_numNonZeroValues);
                if (output_seqLen != seqLen)
                {
                    throw new ApplicationException("CopyVariableValueTo returns incorrect sequenceLength");
                }
                if (!output_colsStarts.SequenceEqual(colStarts))
                {
                    throw new ApplicationException("CopyVariableValueTo returns incorrect colStarts");
                }
                if (!output_rowIndices.SequenceEqual(rowIndices))
                {
                    throw new ApplicationException("CopyVariableValueTo returns incorrect rowIndices");
                }
                if (!output_nonZeroValues.SequenceEqual(nonZeroValues))
                {
                    throw new ApplicationException("CopyVariableValueTo returns incorrect nonZeroValues");
                }
                if (output_numNonZeroValues != nonZeroValues.Count())
                {
                    throw new ApplicationException("CopyVariableValueTo returns incorrect numNonZeroValues");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }
    }
}
