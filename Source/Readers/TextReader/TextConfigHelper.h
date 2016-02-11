//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include "Config.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A helper class for text specific parameters.
// A simple wrapper around CNTK ConfigParameters.
class TextConfigHelper
{
public:
    explicit TextConfigHelper(const ConfigParameters& config);

    // Get all streams that are specified in the configuration.
    std::vector<StreamDescriptionPtr> GetStreams() const;

    // Get full path to the input file.
    std::string GetFilepath() const;

    int GetCpuThreadCount() const
    {
        return m_cpuThreadCount;
    }

    bool ShouldRandomize() const
    {
        return m_randomize;
    }

private:
    TextConfigHelper(const TextConfigHelper&) = delete;
    TextConfigHelper& operator=(const TextConfigHelper&) = delete;

    std::string m_filepath;
    std::vector<StreamDescriptionPtr> m_streams;
    int m_cpuThreadCount;
    bool m_randomize;
};

typedef std::shared_ptr<TextConfigHelper> TextConfigHelperPtr;
} } }
