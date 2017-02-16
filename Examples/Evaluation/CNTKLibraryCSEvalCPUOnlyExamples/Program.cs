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

            // Evalaute a single image.
            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);

            // Evaluate a batch of images
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);

            // Evaluate a single sequence with one-hot vector
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.CPUDevice);

            // Evalaute a batch of variable length sequences with one-hot vector
            CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.CPUDevice);

            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
