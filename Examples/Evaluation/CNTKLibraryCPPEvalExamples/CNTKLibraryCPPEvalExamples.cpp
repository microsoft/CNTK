//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPevalExamples.cpp : Sample application shows how to evaluate a model using CNTK V2 API. 
//

#include <stdio.h>

// define GPU_AVAILABLE, if you want to run evaluation on a GPU device. You also need CNTK GPU binaries.
// undefine GPU_AVAILABLE, if you want to run evaluation on a CPU device.
// #define GPU_AVAILABLE

void MultiThreadsEvaluation(bool);

int main()
{

#ifdef GPU_AVAILABLE
    fprintf(stderr, "\n##### Run eval on GPU device. #####\n");
    MultiThreadsEvaluation(true);
#else
    fprintf(stderr, "\n##### Run eval on CPU device. #####\n");
    MultiThreadsEvaluation(false);
#endif

    fprintf(stderr, "Evaluation complete.\n");

    fflush(stderr);
}
