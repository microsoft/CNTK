//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamplesTest.cpp : Test application for CNTKLibraryCPPEvalExamples, both for CPUOnly and GPU.
//

#include <stdio.h>

void MultiThreadsEvaluation(bool);
bool ShouldRunOnCpu();
bool ShouldRunOnGpu();
void KevinPan(bool isGPUAvailable);

int main()
{
    if (ShouldRunOnGpu())
    {
        fprintf(stderr, "\n##### Test CNTKLibraryCPPEvalExamples on GPU device. #####\n");
        KevinPan(true);
        MultiThreadsEvaluation(true);
    }

    if (ShouldRunOnCpu())
    {

        fprintf(stderr, "\n##### Test CNTKLibraryCPPEvalExamples on CPU device. #####\n");
        KevinPan(false);
        MultiThreadsEvaluation(false);
    }

    fprintf(stderr, "Evaluation complete.\n");

    fflush(stderr);
}
