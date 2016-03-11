//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include "Config.h"
#include "Descriptors.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A helper class for text specific parameters.
// A simple wrapper around CNTK ConfigParameters.
class TextConfigHelper
{
public:
    explicit TextConfigHelper(const ConfigParameters& config);

    // Get all input streams that are specified in the configuration.
    const std::vector<StreamDescriptor>& GetStreams() const;

    // Get full path to the input file.
    const std::wstring& GetFilePath() const;

    int GetCpuThreadCount() const;

    bool ShouldRandomize() const;

    ElementType GetElementType() const;

    bool ShouldSkipSequenceIds() const;

    unsigned int GetMaxAllowedErrors() const;

    unsigned int GetTraceLevel() const;

    size_t GetChunkSize() const;

    unsigned int GetNumChunksToCache() const;
    
    void ParseStreamConfig(const ConfigParameters& config, std::vector<StreamDescriptor>& streams);

    DISABLE_COPY_AND_MOVE(TextConfigHelper);

private:
    std::wstring m_filepath;
    std::vector<StreamDescriptor> m_streams;
    int m_cpuThreadCount;
    bool m_randomize;
    ElementType m_elementType;
    bool m_skipSequenceIds;
    unsigned int m_maxErrors;
    unsigned int m_traceLevel;
    size_t m_chunkSizeBytes; // chunks size in bytes
    unsigned int m_chunkCacheSize; // number of chunks to keep in the memory
};

} } }
