//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs -- Example for using CNTK Library C# Eval GPU Nuget Package.
//

using System;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======== Evaluate model using C# GPU Build ========");

            Console.WriteLine(" ====== Run evaluation on CPU =====");

            // Evalaute a single image.
            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);

            // Evaluate a batch of images
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);

            // Evaluate a single sequence with one-hot vector
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.CPUDevice);

            // Evalaute a batch of variable length sequences with one-hot vector
            CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.CPUDevice);

            // Use GPU for evaluation.
            Console.WriteLine(" ====== Run evaluation on GPU =====");
            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.GPUDevice(0));

            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
