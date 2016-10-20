//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalV2Client.cpp : Sample application shows how to evaluate a model using CNTK V2 API. 
//

#include <stdio.h>

// define CPUONLY, if you want to run evaluation on a CPU device.
// undefine CPUONLY, if you want to run evaluation on a GPU device. You also need CNTK GPU binaries.
#define CPUONLY

void MultiThreadsEvaluation(bool);

int main()
{

#ifndef CPUONLY
    fprintf(stderr, "\n##### Run eval on GPU device. #####\n");
    MultiThreadsEvaluation(true);
#else
    fprintf(stderr, "\n##### Run eval on CPU device. #####\n");
    MultiThreadsEvaluation(false);
#endif

}
