//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <string>
#include "TextConfigHelper.h"
#include "StringUtil.h"

using std::string;
using std::pair;
using std::vector;

namespace Microsoft { namespace MSR { namespace CNTK {

    TextConfigHelper::TextConfigHelper(const ConfigParameters& config)
    {
        ParseStreamConfig(config("input"), m_inputStreams);

        ParseStreamConfig(config("output"), m_outputStreams);

        m_filepath = config("file");

        string rand = config("randomize", "auto");

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

        //TODO: chunk_size (if ever need chunks).
    }

    void TextConfigHelper::ParseStreamConfig(const ConfigParameters& config, std::vector<StreamDescriptor>& streams)
    {
        StreamId id = 0;
        for (const pair<string, ConfigParameters>& section : config)
        {
            ConfigParameters input = section.second;
            const string& name = section.first;

            if (!input.ExistsCurrent("dim") || !input.ExistsCurrent("storage")) {
                RuntimeError("An input section %s does not specify of the required parameters"
                    "(dim, storage).", name.c_str());
            }

            StreamDescriptor stream;
            stream.m_id = id++;
            stream.m_name = msra::strfun::utf16(name);
            stream.m_sampleSize = input("dim");
            string type = input("storage");

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
                RuntimeError("'storage' parameter must be set either to 'dense' or 'sparse'");
            }

            // alias is optional
            if (input.ExistsCurrent("alias")) {
                stream.m_alias = input("alias");
            }
            else {
                stream.m_alias = name;
            }

            // TODO: add double support (with an optional "precision" parameter)
            stream.m_elementType = ElementType::tfloat;
            streams.push_back(stream);
        }
    }

    const string& TextConfigHelper::GetFilepath() const
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

    const vector<StreamDescriptor>& TextConfigHelper::GetInputStreams() const
    {
        return m_inputStreams;
    }

    const vector<StreamDescriptor>& TextConfigHelper::GetOutputStreams() const
    {
        return m_outputStreams;
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

}}}
