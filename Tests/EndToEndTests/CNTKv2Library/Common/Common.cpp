// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory>
#include "CNTKLibrary.h"

using namespace CNTK;

namespace CNTK
{
    class MinibatchSource;
    typedef std::shared_ptr<MinibatchSource> MinibatchSourcePtr;
}

#ifdef _MSC_VER
#include "Windows.h"

// In case of asserts in debug mode, print the message into stderr and throw exception
int HandleDebugAssert(int /* reportType */,
                      char *message,
                      int *)
{
    fprintf(stderr, "C-Runtime error: %s\n", message);
    RaiseFailFastException(0, 0, FAIL_FAST_GENERATE_EXCEPTION_ADDRESS);
    return TRUE;
}
#endif

bool ShouldRunOnCpu()
{
  const char* p = getenv("TEST_DEVICE");

  return (p == nullptr) || !strcmp(p, "cpu");
}

bool ShouldRunOnGpu()
{
#ifdef CPUONLY
  return false;
#else
  const char* p = getenv("TEST_DEVICE");
  return (p == nullptr) || !strcmp(p, "gpu");
#endif
}

bool Is1bitSGDAvailable()
{
    static bool is1bitSGDAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
        const char* p = getenv("TEST_1BIT_SGD");

        // Check the environment variable TEST_1BIT_SGD to decide whether to run 1-bit SGD tests.
        if (p != nullptr && 0 == strcmp(p, "0"))
        {
            is1bitSGDAvailable = false;
        }
        else
        {
            is1bitSGDAvailable = true;
        }
        isInitialized = true;
    }

    return is1bitSGDAvailable;
}

MinibatchSourceConfig GetHTKMinibatchSourceConfig(size_t featureDim, size_t numOutputClasses, size_t epochSize, bool randomize = true)
{
    auto featuresFilePath = L"glob_0000.scp";
    auto labelsFilePath = L"glob_0000.mlf";
    auto labelMappingFile = L"state.list";

    Deserializer featureDeserializer = HTKFeatureDeserializer({ HTKFeatureConfiguration(L"features", featuresFilePath, featureDim, 0, 0, false) });
    Deserializer labelDeserializer = HTKMLFDeserializer(L"labels", labelMappingFile, numOutputClasses, { labelsFilePath });

    MinibatchSourceConfig config({ featureDeserializer, labelDeserializer }, randomize);
    config.maxSamples = epochSize;
    return config;
}
