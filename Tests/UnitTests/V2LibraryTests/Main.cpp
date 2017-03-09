//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

using namespace CNTK;

void NDArrayViewTests();
void ValueTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void TrainCifarResnet();
void BlockTests();
void FunctionTests();
void TrainLSTMSequenceClassifer();
void SerializationTests();
void LearnerTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifer();
void DeviceSelectionTests();
void MultiThreadsEvaluation(bool);
void MinibatchSourceTests();
void UserDefinedFunctionTests();
void LoadLegacyModelTests();

int main()
{
#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

#ifndef CPUONLY
    fprintf(stderr, "Run tests on %s device using GPU build.\n", IsGPUAvailable() ? "GPU" : "CPU");
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

    // Note: Run the device selection tests first since later tests
    // may interfere with device selection by freezing default device
    DeviceSelectionTests();

    NDArrayViewTests();
    ValueTests();
    TensorTests();
    FunctionTests();
    BlockTests();

    FeedForwardTests();
    RecurrentFunctionTests();
    UserDefinedFunctionTests();

    SerializationTests();
    LoadLegacyModelTests();

    LearnerTests();

    TrainerTests();
    TrainCifarResnet();
    TrainLSTMSequenceClassifer();

    TrainSequenceToSequenceTranslator();
    TrainTruncatedLSTMAcousticModelClassifer();

    MinibatchSourceTests();

    MultiThreadsEvaluation(IsGPUAvailable());

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif
}
