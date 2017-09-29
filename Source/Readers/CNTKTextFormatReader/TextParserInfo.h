#pragma once

#include "CNTKLibrary.h"

namespace CNTK {

struct TextParserInfo {
    TextParserInfo(const std::wstring& filename, const unsigned int& traceLevel, const unsigned int& numAllowedErrors, const size_t& maxAliasLength,
       const bool& useMaxAsSequenceLength, const std::map<std::string, size_t>& aliasToIdMap, const std::vector<StreamInformation>& streamInfos) :
        m_filename(filename),
        m_traceLevel(traceLevel),
        m_numAllowedErrors(numAllowedErrors),
        m_useMaxAsSequenceLength(useMaxAsSequenceLength),
        m_maxAliasLength(maxAliasLength),
        m_aliasToIdMap(aliasToIdMap),
        m_streamInfos(streamInfos)
    {
    }

    std::wstring m_filename;
    unsigned int m_traceLevel;
    unsigned int m_numAllowedErrors;

    // Indicates if the sequence length is computed as the maximum 
    // of number of samples across all streams (inputs).
    bool m_useMaxAsSequenceLength;

    size_t m_maxAliasLength;
    std::map<std::string, size_t> m_aliasToIdMap;

    std::vector<StreamInformation> m_streamInfos;
};

}