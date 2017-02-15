//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"

#include <iostream>
#include <cstdio>

using namespace CNTK;
using namespace std::placeholders;

void TrainCifarResnet();
void TrainLSTMSequenceClassifer();
void MNISTClassifierTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifer();
void TestFrameMode();

int main(int argc, char *argv[])
{
#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

#ifndef CPUONLY
    fprintf(stderr, "Run tests using GPU build.\n");
#else
    fprintf(stderr, "Run tests using CPU-only build.\n");
#endif

    if (argc > 2)
    {
        if (argc == 3 && !std::string(argv[1]).compare("Distribution")) {
            {
                auto communicator = MPICommunicator();
                std::string logFilename = argv[2] + std::to_string(communicator->CurrentWorker().m_globalRank);
                auto result = freopen(logFilename.c_str(), "w", stdout);
                if (result == nullptr)
                {
                    fprintf(stderr, "Could not redirect stdout.\n");
                    return -1;
                }
            }

            TestFrameMode();

            std::string testsPassedMsg = "\nCNTKv2Library-Distribution tests: Passed\n";

            printf("%s", testsPassedMsg.c_str());

            fflush(stdout);
            DistributedCommunicator::Finalize();
            fclose(stdout);
            return 0;
        }
        else
        {
            fprintf(stderr, "Wrong number of arguments.\n");
            return -1;
        }
    }

    std::string testName(argv[1]);

    if (!testName.compare("CifarResNet"))
    {
        if (IsGPUAvailable())
        {
            fprintf(stderr, "Run test on a GPU device.\n");
            TrainCifarResnet();
        }
        else
        {
            fprintf(stderr, "Cannot run TrainCifarResnet test on a CPU device.\n");
        }
    }
    else if (!testName.compare("LSTMSequenceClassifier"))
    {
        TrainLSTMSequenceClassifer();
    }
    else if (!testName.compare("MNISTClassifier"))
    {
        MNISTClassifierTests();
    }
    else if (!testName.compare("SequenceToSequence"))
    {
        TrainSequenceToSequenceTranslator();
    }
    else if (!testName.compare("TruncatedLSTMAcousticModel"))
    {
        TrainTruncatedLSTMAcousticModelClassifer();
    }
    else
    {
        fprintf(stderr, "End to end test not found.\n");
        return -1;
    }

    std::string testsPassedMsg = "\nCNTKv2Library-" + testName + " tests: Passed\n";

    fprintf(stderr, "%s", testsPassedMsg.c_str());
    fflush(stderr);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif
}
