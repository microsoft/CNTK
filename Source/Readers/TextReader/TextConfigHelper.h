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
    const std::vector<StreamDescriptor>& GetInputStreams() const;

    // Get all input streams that are specified in the configuration.
    const std::vector<StreamDescriptor>& GetOutputStreams() const;

    // Get full path to the input file.
    const std::string& GetFilepath() const;

    int GetCpuThreadCount() const;

    bool ShouldRandomize() const;

    bool ShouldSkipSequenceIds() const;

    unsigned int GetMaxAllowedErrors() const;

    unsigned int GetTraceLevel() const;

    void ParseStreamConfig(const ConfigParameters& config, std::vector<StreamDescriptor>& streams);

private:
    TextConfigHelper(const TextConfigHelper&) = delete;
    TextConfigHelper& operator=(const TextConfigHelper&) = delete;

    std::string m_filepath;
    std::vector<StreamDescriptor> m_inputStreams;
    std::vector<StreamDescriptor> m_outputStreams;
    int m_cpuThreadCount;
    bool m_randomize;
    bool m_skipSequenceIds;
    unsigned int m_maxErrors;
    unsigned int m_traceLevel;

};

} } }
