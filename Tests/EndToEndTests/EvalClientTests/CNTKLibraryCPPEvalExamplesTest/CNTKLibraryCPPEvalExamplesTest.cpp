//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamplesTest.cpp : Test application for CNTKLibraryCPPEvalExamples, both for CPUOnly and GPU.
//

#include <stdio.h>

void MultiThreadsEvaluation(bool);
bool IsGPUAvailable();

int main()
{
    if (IsGPUAvailable())
    {
        fprintf(stderr, "\n##### Test CNTKLibraryCPPEvalExamples on GPU device. #####\n");
        MultiThreadsEvaluation(true);
    }
    else
    {
        fprintf(stderr, "\n##### Test CNTKLibraryCPPEvalExamples on CPU device. #####\n");
        MultiThreadsEvaluation(false);
    }

    fprintf(stderr, "Evaluation complete.\n");

    fflush(stderr);
}