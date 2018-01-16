//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs -- Example for using CNTK Library C# Eval GPU Nuget Package.
//

using System;
using System.Threading.Tasks;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======== Evaluate model using C# GPU Build ========");

            Console.WriteLine(" ====== Run evaluation on CPU =====");

            // Evaluate a single image.
            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);

            // Evaluate a batch of images.
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);

            // Evaluate multiple sample requests in parallel.
            CNTKLibraryManagedExamples.EvaluateMultipleImagesInParallelAsync(DeviceDescriptor.CPUDevice).Wait();

            // Evaluate an image asynchronously.
            Task evalTask = CNTKLibraryManagedExamples.EvaluationSingleImageAsync(DeviceDescriptor.CPUDevice);
            evalTask.Wait();

            // Evaluate a single sequence using one-hot vector input.
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.CPUDevice);

            // Evaluate a batch of variable length sequences with one-hot vector input.
            CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.CPUDevice);

            // Evaluate a sequence using sparse input.
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingSparse(DeviceDescriptor.CPUDevice);

            // Load model from memory buffer.
            CNTKLibraryManagedExamples.LoadModelFromMemory(DeviceDescriptor.CPUDevice);

            // Use GPU for evaluation.
            Console.WriteLine(" ====== Run evaluation on GPU =====");
            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.GPUDevice(0));

            // Evaluate an image asynchronously
            evalTask = CNTKLibraryManagedExamples.EvaluationSingleImageAsync(DeviceDescriptor.GPUDevice(0));
            evalTask.Wait();

            CNTKLibraryManagedExamples.EvaluateMultipleImagesInParallelAsync(DeviceDescriptor.GPUDevice(0)).Wait();
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.EvaluationSingleSequenceUsingSparse(DeviceDescriptor.GPUDevice(0));
            CNTKLibraryManagedExamples.LoadModelFromMemory(DeviceDescriptor.GPUDevice(0));

            // Evaluate intermediate layer.
            CNTKLibraryManagedExamples.EvaluateIntermediateLayer(DeviceDescriptor.GPUDevice(0));

            // Evaluate combined outputs.
            CNTKLibraryManagedExamples.EvaluateCombinedOutputs(DeviceDescriptor.GPUDevice(0));

            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
