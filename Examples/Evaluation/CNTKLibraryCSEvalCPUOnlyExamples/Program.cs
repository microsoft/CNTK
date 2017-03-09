//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs -- Example for using CNTK Library C# Eval CPUOnly Nuget Package.
//

using System;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======== Evaluate model using C# CPUOnly Build ========");

            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);
            
            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
