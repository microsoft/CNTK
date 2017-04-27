//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include <map>
#include "Config.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A helper class for binary specific parameters.
// A simple wrapper around CNTK ConfigParameters.
class BinaryConfigHelper
{
public:
    explicit BinaryConfigHelper(const ConfigParameters& config);

    // Get all input streams that are specified in the configuration.
    const std::map<std::wstring, std::wstring>& GetRename() const { return m_streams; }

    // Get full path to the input file.
    const wstring& GetFilePath() const { return m_filepath; }

    size_t GetRandomizationWindow() const { return m_randomizationWindow; }

    bool UseSampleBasedRandomizationWindow() const { return m_sampleBasedRandomizationWindow; }

    unsigned int GetTraceLevel() const { return m_traceLevel; }

    bool ShouldKeepDataInMemory() const { return m_keepDataInMemory; }

    ElementType GetElementType() const { return m_elementType; }

    DISABLE_COPY_AND_MOVE(BinaryConfigHelper);

private:
    std::wstring m_filepath;
    std::map<std::wstring, std::wstring> m_streams;
    ElementType m_elementType;
    size_t m_randomizationWindow;
    // Specifies how to interpret randomization window, if true randomization window == number of samples, else 
    // randomization window = number of chunks (default).
    bool m_sampleBasedRandomizationWindow;
    unsigned int m_traceLevel;
    bool m_keepDataInMemory; // if true the whole dataset is kept in memory
};

} } }
