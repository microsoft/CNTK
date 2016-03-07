//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "TextConfigHelper.h"
#include "StringUtil.h"

using std::string;
using std::wstring;
using std::pair;
using std::vector;
using std::map;

namespace Microsoft { namespace MSR { namespace CNTK {

    TextConfigHelper::TextConfigHelper(const ConfigParameters& config)
    {

        if (!config.ExistsCurrent(L"input")) 
        {
            RuntimeError("CNTKTextFormatReader configuration does not contain input section");
        }

        const ConfigParameters& input = config(L"input");

        if (input.empty())
        {
            RuntimeError("CNTKTextFormatReader configuration contains an empty input section");
        }

        StreamId id = 0;
        map<string, wstring> aliasToInputMap;
        for (const pair<string, ConfigParameters>& section : input)
        {
            ConfigParameters input = section.second;
            const wstring& name = msra::strfun::utf16(section.first);

            if (!input.ExistsCurrent(L"dim") || !input.ExistsCurrent(L"format")) {
                RuntimeError("Input section for input '%ls' does not specify all the required parameters, "
                    "\"dim\" and \"format\".", name.c_str());
            }

            StreamDescriptor stream;
            stream.m_id = id++;
            stream.m_name = name;
            stream.m_sampleSize = input(L"dim");
            string type = input(L"format");

            if (AreEqualIgnoreCase(type, "dense"))
            {
                stream.m_storageType = StorageType::dense;
            }
            else if (AreEqualIgnoreCase(type, "sparse"))
            {
                stream.m_storageType = StorageType::sparse_csc;
            }
            else
            {
                RuntimeError("'format' parameter must be set either to 'dense' or 'sparse'");
            }

            // alias is optional
            if (input.ExistsCurrent(L"alias")) {
                stream.m_alias = input(L"alias");
                if (stream.m_alias.empty()) 
                {
                    RuntimeError("Alias value for input '%ls' is empty", name.c_str());
                }
            }
            else 
            {
                stream.m_alias = section.first;
            }

            if (aliasToInputMap.find(stream.m_alias) != aliasToInputMap.end()) 
            {
                RuntimeError("Alias %s is already mapped to input %ls.", 
                    stream.m_alias, aliasToInputMap[stream.m_alias]);
            }
            else
            {
                aliasToInputMap[stream.m_alias] = stream.m_name;
            }

            // TODO: add double support (with an optional "precision" parameter)
            stream.m_elementType = ElementType::tfloat;
            m_streams.push_back(stream);
        }



        m_filepath = msra::strfun::utf16(config(L"file"));

        string rand = config(L"randomize", "auto");

        if (AreEqualIgnoreCase(rand, "auto"))
        {
            m_randomize = true;
        }
        else if (AreEqualIgnoreCase(rand, "none"))
        {
            m_randomize = false;
        }
        else
        {
            RuntimeError("'randomize' parameter must be set to 'auto' or 'none'");
        }

        m_cpuThreadCount = config(L"numCPUThreads", 0);
        m_skipSequenceIds = config(L"skipSequenceIds", false);
        m_maxErrors = config(L"maxErrors", 0);
        m_traceLevel = config(L"traceLevel", 0);
        m_chunkSize = config(L"chunkSize", 32 * 1024 * 1024); // 32 MB by default
        m_chunkCacheSize = config(L"chunkCacheSize", 32); // GB of memory in total
    }

    const wstring& TextConfigHelper::GetFilePath() const
    {
        return m_filepath;
    }

    int TextConfigHelper::GetCpuThreadCount() const
    {
        return m_cpuThreadCount;
    }

    bool TextConfigHelper::ShouldRandomize() const
    {
        return m_randomize;
    }

    const vector<StreamDescriptor>& TextConfigHelper::GetStreams() const
    {
        return m_streams;
    }

    bool TextConfigHelper::ShouldSkipSequenceIds() const
    {
        return m_skipSequenceIds;
    }

    unsigned int TextConfigHelper::GetMaxAllowedErrors() const
    {
        return m_maxErrors;
    }

    unsigned int TextConfigHelper::GetTraceLevel() const
    {
        return m_traceLevel;
    }

    int64_t TextConfigHelper::GetChunkSize() const 
    {
        return m_chunkSize;
    }

    unsigned int TextConfigHelper::GetNumChunksToCache() const
    {
        return m_chunkCacheSize;
    }
}}}
