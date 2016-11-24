//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"
#include <cstdio>

using namespace CNTK;
using namespace std::placeholders;

void TestFrameMode();

int main(int argc, char* argv[])
{
#if defined(_MSC_VER)
    // in case of asserts in debug mode, print the message into stderr and throw exception
    if (_CrtSetReportHook2(_CRT_RPTHOOK_INSTALL, HandleDebugAssert) == -1) {
        fprintf(stderr, "_CrtSetReportHook2 failed.\n");
        return -1;
    }
#endif

    if (argc != 2)
    {
        fprintf(stderr, "Expecting a log file parameter.\n");
        return -1; // Unexpected number of parameters given.
    }

    {
        auto communicator = MPICommunicator();
        std::string logFilename = argv[1] + std::to_string(communicator->CurrentWorker().m_globalRank);
        auto result = freopen(logFilename.c_str(), "w", stdout);
        if (result == nullptr)
        {
            fprintf(stderr, "Could not redirect stdout.\n");
            return -1;
        }
    }

    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::SetAutomaticUnpackingOfPackedValues(/*disable =*/ true);

    TestFrameMode();

    printf("\nCNTKv2LibraryDistribution tests: Passed\n");
    fflush(stdout);

#if defined(_MSC_VER)
    _CrtSetReportHook2(_CRT_RPTHOOK_REMOVE, HandleDebugAssert);
#endif

    DistributedCommunicator::Finalize();

    fclose(stdout);
    return 0;
}