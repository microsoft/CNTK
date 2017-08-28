//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalCPUOnlyExamples.cpp: Model evaluation using CNTK V2 C++ API on CPUOnly device.
//

#include <stdio.h>
#include "CNTKLibrary.h"

void EvaluationSingleSampleUsingDense(const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationBatchUsingDense(const wchar_t*, const CNTK::DeviceDescriptor&);
void ParallelEvaluationExample(const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingOneHot(const wchar_t*, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationBatchOfSequencesUsingOneHot(const wchar_t*, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingSparse(const wchar_t*, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluateIntermediateLayer(const wchar_t*, const CNTK::DeviceDescriptor& device);
void EvaluateCombinedOutputs(const wchar_t*, const CNTK::DeviceDescriptor& device);

int main()
{
    // The resnet20.dnn model is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    const wchar_t* resnet20Model = L"resnet20.dnn";

    // The atis.dnn model is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
    // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
    const wchar_t* atisModel = L"atis.dnn";

    // The query.wl is the vacabulary file used by the ATIS model. It is available in <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript.
    const wchar_t* vocabularyFile = L"query.wl";

    // The slots.wl is the label file used by the ATIS model. It is available in <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript.
    const wchar_t* labelFile = L"slots.wl";

    printf("\n##### Run CNTKLibraryCPPEvalCPUOnlyExamples on CPU. #####\n");

    // Evaluate a single image with reset20_cifar model.
    EvaluationSingleSampleUsingDense(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate batch of images with reset20_cifar model.
    EvaluationBatchUsingDense(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate users requests in parallel with reset20_cifar model.
    ParallelEvaluationExample(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate a single sequence with ATIS model.
    EvaluationSingleSequenceUsingOneHot(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate batch of sequences with ATIS model.
    EvaluationBatchOfSequencesUsingOneHot(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate a sequence using sparse input with ATIS model.
    EvaluationSingleSequenceUsingSparse(atisModel, vocabularyFile, labelFile, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate intermediate layer
    EvaluateIntermediateLayer(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

    // Evaluate combined outputs
    EvaluateCombinedOutputs(resnet20Model, CNTK::DeviceDescriptor::CPUDevice());

    printf("Evaluation complete.\n");
}
