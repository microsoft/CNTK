//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamplesTest.cpp : Test application for CNTKLibraryCPPEvalExamples, both for CPUOnly and GPU.
//

#include <stdio.h>

#include "CNTKLibrary.h"

void MultiThreadsEvaluationTests(const wchar_t*, bool);
void EvaluationSingleSampleUsingDense(const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationBatchUsingDense(const wchar_t*, const CNTK::DeviceDescriptor&);
void ParallelEvaluationExample(const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingOneHot(const wchar_t*, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationBatchOfSequencesUsingOneHot(const wchar_t*, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingSparse(const wchar_t*, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluateIntermediateLayer(const wchar_t*, const CNTK::DeviceDescriptor& device);
void EvaluateCombinedOutputs(const wchar_t*, const CNTK::DeviceDescriptor& device);
bool ShouldRunOnCpu();
bool ShouldRunOnGpu();

int main()
{
    const wchar_t* oneHiddenModel = L"01_OneHidden.model";
    const wchar_t* resnet20Model = L"resnet20.dnn";
    const wchar_t* atisModel = L"atis.dnn";
    const wchar_t* vocabularyFile = L"query.wl";
    const wchar_t* labelFile = L"slots.wl";

    if (ShouldRunOnGpu())
    {
        printf("\n##### Test CPPEval samples on GPU device. #####\n");
        EvaluationSingleSampleUsingDense(resnet20Model, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationBatchUsingDense(resnet20Model, CNTK::DeviceDescriptor::GPUDevice(0));
        ParallelEvaluationExample(resnet20Model, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationSingleSequenceUsingOneHot(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationBatchOfSequencesUsingOneHot(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationSingleSequenceUsingSparse(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::GPUDevice(0));

        printf("\n##### Test MultiThreadsEvaluation on GPU device. #####\n");
        MultiThreadsEvaluationTests(oneHiddenModel, true);
        EvaluateIntermediateLayer(resnet20Model, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluateCombinedOutputs(resnet20Model, CNTK::DeviceDescriptor::GPUDevice(0));
    }

    if (ShouldRunOnCpu())
    {
        printf("\n##### Test CPPEval samples on CPU device. #####\n");
        EvaluationSingleSampleUsingDense(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationBatchUsingDense(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());
        ParallelEvaluationExample(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationSingleSequenceUsingOneHot(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationBatchOfSequencesUsingOneHot(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationSingleSequenceUsingSparse(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::CPUDevice());

        printf("\n##### Test MultiThreadsEvaluation CPU device. #####\n");
        MultiThreadsEvaluationTests(oneHiddenModel, false);
        EvaluateIntermediateLayer(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());
        EvaluateCombinedOutputs(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());
    }

    printf("Evaluation complete.\n");
    fflush(stdout);
}
