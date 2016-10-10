//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void TrainCifarResnet();
void FunctionTests();
void TrainLSTMSequenceClassifer();
void SerializationTests();
void LearnerTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifer();
void DeviceSelectionTests();
void MultiThreadsEvaluation(bool);

int main()
{

#ifndef CPUONLY
    if (IsGPUAvailable())
    {
        fprintf(stderr, "Run tests on GPU device using GPU build.\n");
    }
    else
    {
        fprintf(stderr, "Run tests on CPU device using GPU build.\n");
    }
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::DisableAutomaticUnpackingOfPackedValues();

    NDArrayViewTests();
    TensorTests();
    FunctionTests();

    FeedForwardTests();
    RecurrentFunctionTests();

    SerializationTests();
    LearnerTests();

    TrainerTests();
    TrainCifarResnet();
    TrainLSTMSequenceClassifer();

    TrainSequenceToSequenceTranslator();
    TrainTruncatedLSTMAcousticModelClassifer();

    MultiThreadsEvaluation(IsGPUAvailable());

    fprintf(stderr, "Test device selection API\n");
    DeviceSelectionTests();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
