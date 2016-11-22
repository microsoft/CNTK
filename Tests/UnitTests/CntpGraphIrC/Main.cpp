//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include <functional>


using namespace CNTK;

using namespace std;

void MultiThreadsEvaluation();


int main()
{
#ifndef CPUONLY
#error "must use CPU Only"
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

    MultiThreadsEvaluation();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
