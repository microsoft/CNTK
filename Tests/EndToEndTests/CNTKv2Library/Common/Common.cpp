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

bool IsGPUAvailable()
{
    static bool isGPUDeviceAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
#ifndef CPUONLY
        const char* p = getenv("TEST_DEVICE");

        // Check the environment variable TEST_DEVICE to decide whether to run on a CPU-only device.
        if (p != nullptr && !strcmp(p, "cpu"))
        {
            isGPUDeviceAvailable = false;
        }
        else
        {
            isGPUDeviceAvailable = true;
        }
#else
        isGPUDeviceAvailable = false;
#endif
        isInitialized = true;
    }

    return isGPUDeviceAvailable;
}

bool Is1bitSGDAvailable()
{
    static bool is1bitSGDAvailable;
    static bool isInitialized = false;

    if (!isInitialized)
    {
        const char* p = getenv("TEST_1BIT_SGD");

        // Check the environment variable TEST_1BIT_SGD to decide whether to run on a CPU-only device.
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

 MinibatchSourcePtr CreateHTKMinibatchSource(size_t featureDim, size_t numOutputClasses, const Dictionary& readModeConfig, size_t epochSize, bool randomize = true)
{
    auto featuresFilePath = L"glob_0000.scp";
    auto labelsFilePath = L"glob_0000.mlf";
    auto labelMappingFile = L"state.list";

    Dictionary featuresStreamConfig;
    featuresStreamConfig[L"dim"] = featureDim;
    featuresStreamConfig[L"scpFile"] = featuresFilePath;

    Dictionary featInputStreamsConfig;
    featInputStreamsConfig[L"features"] = featuresStreamConfig;

    Dictionary featDeserializerConfiguration;
    featDeserializerConfiguration[L"type"] = L"HTKFeatureDeserializer";
    featDeserializerConfiguration[L"input"] = featInputStreamsConfig;

    Dictionary labelsStreamConfig;
    labelsStreamConfig[L"dim"] = numOutputClasses;
    labelsStreamConfig[L"mlfFile"] = labelsFilePath;
    labelsStreamConfig[L"labelMappingFile"] = labelMappingFile;
    labelsStreamConfig[L"scpFile"] = featuresFilePath;

    Dictionary labelsInputStreamsConfig;
    labelsInputStreamsConfig[L"labels"] = labelsStreamConfig;

    Dictionary labelsDeserializerConfiguration;
    labelsDeserializerConfiguration[L"type"] = L"HTKMLFDeserializer";
    labelsDeserializerConfiguration[L"input"] = labelsInputStreamsConfig;

    Dictionary minibatchSourceConfiguration;
    if (randomize)
        minibatchSourceConfiguration[L"randomize"] = true;

    minibatchSourceConfiguration[L"epochSize"] = epochSize;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<DictionaryValue>({ featDeserializerConfiguration, labelsDeserializerConfiguration });
    minibatchSourceConfiguration.Add(readModeConfig);

    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
}
