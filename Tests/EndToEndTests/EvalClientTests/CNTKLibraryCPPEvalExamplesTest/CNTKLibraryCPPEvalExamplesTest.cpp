//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamplesTest.cpp : Test application for CNTKLibraryCPPEvalExamples, both for CPUOnly and GPU.
//

#include <stdio.h>

void MultiThreadsEvaluation(const wchar_t* modelFileName, bool);
bool ShouldRunOnCpu();
bool ShouldRunOnGpu();

int main()
{
    const wchar_t* modelFileName = L"01_OneHidden";
    if (ShouldRunOnGpu())
    {
        fprintf(stderr, "\n##### Test CNTKLibraryCPPEvalExamples on GPU device. #####\n");
        MultiThreadsEvaluation(modelFileName, true);
    }

    if (ShouldRunOnCpu())
    {
        fprintf(stderr, "\n##### Test CNTKLibraryCPPEvalExamples on CPU device. #####\n");
        MultiThreadsEvaluation(modelFileName, false);
    }

    fprintf(stderr, "Evaluation complete.\n");

    fflush(stderr);
}
