//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This implements a plain-text deserializer
//

#include "stdafx.h"
#include "DataDeserializer.h"
#include "ReaderConstants.h"
#include "File.h"

#include <memory>

#define let const auto

using namespace CNTK;
using namespace std;

namespace CNTK {

// ===========================================================================
// This class implements a plain-text deserializer, for lines of space-separate tokens
// as needed for many language-modeling and translation tasks.
// ===========================================================================

class PlainTextDeserializerImpl : public DataDeserializer
{
public:
    const struct PlainTextVocabularyConfiguration
    {
        const std::wstring fileName;
        const std::wstring insertAtStart;
        const std::wstring insertAtEnd;
        const std::wstring substituteForUnknown;
    };
    struct PlainTextStreamConfiguration : public StreamInformation
    {
        PlainTextStreamConfiguration(StreamInformation&& streamInfo, vector<wstring>&& fileNames, PlainTextVocabularyConfiguration&& vocabularyConfig) :
            StreamInformation(move(streamInfo)), m_fileNames(move(fileNames)), m_vocabularyConfig(vocabularyConfig)
        { }
        const vector<wstring> m_fileNames;
        const PlainTextVocabularyConfiguration m_vocabularyConfig;
    };

    class PlainTextStream : public PlainTextStreamConfiguration
    {
    public:
        PlainTextStream(const PlainTextStreamConfiguration& streamConfig) : PlainTextStreamConfiguration(streamConfig) { }
    };

    PlainTextDeserializerImpl(const vector<PlainTextDeserializerImpl::PlainTextStreamConfiguration>& streamConfigs,
                              DataType elementType, bool primary, int traceLevel) :
        m_streams(streamConfigs.begin(), streamConfigs.end()), m_elementType(elementType), m_primary(primary), m_traveLevel(traceLevel)
    {
    }

    ///
    /// Gets stream information for all streams this deserializer exposes.
    ///
    virtual std::vector<StreamInformation> StreamInfos() override
    {
        return vector<StreamInformation>(m_streams.begin(), m_streams.end());
    }

    ///
    /// Gets metadata for chunks this deserializer exposes.
    ///
    virtual std::vector<ChunkInfo> ChunkInfos() override
    {
        return std::vector<ChunkInfo>();
    }

    ///
    /// Gets sequence infos for a given a chunk.
    ///
    virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override
    {
        chunkId; result;
    }

    ///
    /// Gets sequence information given the one of the primary deserializer.
    /// Used for non-primary deserializers.
    /// Returns false if the corresponding secondary sequence is not valid.
    ///
    virtual bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo& result) override
    {
        primary; result;
        return true;
    }

    ///
    /// Gets chunk data given its id.
    ///
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override
    {
        chunkId;
        return nullptr;
    }

private:
    vector<PlainTextStream> m_streams;
    const DataType m_elementType;
    const bool m_primary;
    const int m_traveLevel;
};

// factory function to create a PlainTextDeserializer from a V1 config object
shared_ptr<DataDeserializer> CreatePlainTextDeserializer(const ConfigParameters& configParameters, bool primary)
{
    // convert V1 configParameters back to a C++ data structure
    const ConfigParameters& input = configParameters(L"input");
    if (input.empty())
        InvalidArgument("PlainTextDeserializer configuration contains an empty \"input\" section.");

    string precision = configParameters.Find("precision", "float");
    DataType elementType;
    if (precision == "double")
        elementType = DataType::Double;
    else if (precision == "float")
        elementType = DataType::Float;
    else
        RuntimeError("Not supported precision '%s'. Expected 'double' or 'float'.", precision.c_str());

    // enumerate all streams
    vector<PlainTextDeserializerImpl::PlainTextStreamConfiguration> streamConfigs; streamConfigs.reserve(input.size());
    for (const pair<string, ConfigParameters>& section : input)
    {
        ConfigParameters streamConfigParameters = section.second;
        // basic stream info
        StreamInformation streamInfo;
        streamInfo.m_name          = msra::strfun::utf16(section.first);
        streamInfo.m_id            = streamConfigs.size(); // id = index of stream config in stream-config array
        streamInfo.m_storageFormat = StorageFormat::SparseCSC;
        streamInfo.m_elementType   = elementType;
        streamInfo.m_sampleLayout  = NDShape{ streamConfigParameters(L"dim") };
        streamInfo.m_definesMbSize = streamConfigParameters(L"definesMBSize", false);
        // list of files. Wildcard expansion is happening here.
        vector<wstring> fileNames = streamConfigParameters(L"dataFiles", ConfigParameters::Array(Microsoft::MSR::CNTK::stringargvector()));
        vector<wstring> expandedFileNames;
        for (let& fileName : fileNames)
            expand_wildcards(fileName, expandedFileNames);
        // vocabularyConfig
        streamConfigs.emplace_back(PlainTextDeserializerImpl::PlainTextStreamConfiguration
        (
            move(streamInfo),
            move(expandedFileNames),
            {
                streamConfigParameters(L"vocabularyFile"),
                streamConfigParameters(L"insertAtStart", L""),
                streamConfigParameters(L"insertAtEnd", L""),
                streamConfigParameters(L"substituteForUnknown", L"")
            }
        ));
    }

    let traceLevel = configParameters(L"traceLevel", 1);
    //let chunkSizeBytes = configParameters(L"chunkSizeInBytes", g_32MB); // 32 MB by default   --do we need this?

    // construct the deserializer from that
    return make_shared<PlainTextDeserializerImpl>(move(streamConfigs), elementType, primary, traceLevel);
}

}; // namespace
