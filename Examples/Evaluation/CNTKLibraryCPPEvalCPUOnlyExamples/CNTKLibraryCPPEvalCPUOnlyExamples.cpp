//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalCPUOnlyExamples.cpp : Sample application shows how to evaluate a model using CNTK V2 API.
//

#include <stdio.h>

void MultiThreadsEvaluation(const wchar_t*, bool);

int main()
{
    const wchar_t* modelFileName = L"01_OneHidden.model";
    fprintf(stderr, "\n##### Run CNTKLibraryCPPEvalCPUOnlyExamples on CPU. #####\n");
    MultiThreadsEvaluation(modelFileName, false);

    fprintf(stderr, "Evaluation complete.\n");
    fflush(stderr);
}
