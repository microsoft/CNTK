//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <limits>
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "SequenceData.h"
#include "../HTKMLFReader/htkfeatio.h"
#include "../HTKMLFReader/msra_mgram.h"
#include "latticearchive.h"
#include "StringUtil.h"


#undef max // max is defined in minwindef.h

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

static float s_oneFloat = 1.0;
static double s_oneDouble = 1.0;

// Currently we only have a single mlf chunk that contains a vector of all labels.
// TODO: In the future MLF should be converted to a more compact format that is amenable to chunking.
class MLFDataDeserializer::MLFChunk : public Chunk
{
    MLFDataDeserializer* m_parent;
public:
    MLFChunk(MLFDataDeserializer* parent) : m_parent(parent)
    {}

    virtual void GetSequence(size_t sequenceId, vector<SequenceDataPtr>& result) override
    {
        m_parent->GetSequenceById(sequenceId, result);
    }
};

// Inner class for an utterance.
struct MLFUtterance : SequenceDescription
{
    size_t m_sequenceStart;
};

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : DataDeserializerBase(primary)
{
    // TODO: This should be read in one place, potentially given by SGD.
    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    // MLF cannot control chunking.
    if (primary)
    {
        LogicError("Mlf deserializer does not support primary mode - it cannot control chunking.");
    }

    std::wstring precision = cfg(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? ElementType::tfloat : ElementType::tdouble;

    ConfigParameters input = cfg("input");
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    size_t dimension = config.GetLabelDimension();

    m_withPhoneBoundaries = streamConfig(L"phoneBoundaries", false);
    if (m_frameMode && m_withPhoneBoundaries)
        LogicError("frameMode and phoneBoundaries are not supposed to be used together.");
    wstring labelMappingFile = streamConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, dimension);
    InitializeStream(inputName, dimension);
}

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& labelConfig, const wstring& name)
    : DataDeserializerBase(false)
{
    // The frame mode is currently specified once per configuration,
    // not in the configuration of a particular deserializer, but on a higher level in the configuration.
    // Because of that we are using find method below.
    m_frameMode = labelConfig.Find("frameMode", "true");

    ConfigHelper config(labelConfig);

    config.CheckLabelType();
    size_t dimension = config.GetLabelDimension();

    if (dimension > numeric_limits<IndexType>::max())
    {
        RuntimeError("Label dimension (%" PRIu64 ") exceeds the maximum allowed "
            "value (%" PRIu64 ")\n", dimension, (size_t)numeric_limits<IndexType>::max());
    }

    std::wstring precision = labelConfig(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? ElementType::tfloat : ElementType::tdouble;

    m_withPhoneBoundaries = labelConfig(L"phoneBoundaries", "false");

    wstring labelMappingFile = labelConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, dimension);
    InitializeStream(name, dimension);
}

// Currently we create a single chunk only.
void MLFDataDeserializer::InitializeChunkDescriptions(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath, size_t dimension)
{
    // TODO: Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files

    // TODO: currently we do not use symbol and word tables.
    const msra::lm::CSymbolSet* wordTable = nullptr;
    unordered_map<const char*, int>* symbolTable = nullptr;
    vector<wstring> mlfPaths = config.GetMlfPaths();

    // TODO: Currently we still use the old IO module. This will be refactored later.
    const double htkTimeToFrame = 100000.0; // default is 10ms
    msra::asr::htkmlfreader<msra::asr::htkmlfentry, msra::lattices::lattice::htkmlfwordsequence> labels(mlfPaths, set<wstring>(), stateListPath, wordTable, symbolTable, htkTimeToFrame);

    // Make sure 'msra::asr::htkmlfreader' type has a move constructor
    static_assert(
        is_move_constructible<
        msra::asr::htkmlfreader<msra::asr::htkmlfentry,
        msra::lattices::lattice::htkmlfwordsequence >> ::value,
        "Type 'msra::asr::htkmlfreader' should be move constructible!");

    MLFUtterance description;
    size_t numClasses = 0;
    size_t totalFrames = 0;

    // TODO resize m_keyToSequence with number of IDs from string registry
    for (const auto& l : labels)
    {
        auto key = msra::strfun::utf8(l.first);
        if (!corpus->IsIncluded(key))
            continue;

        size_t id = corpus->KeyToId(key);
        description.m_key.m_sequence = id;

        const auto& utterance = l.second;
        description.m_sequenceStart = m_classIds.size();
        uint32_t numberOfFrames = 0;

        vector<size_t> sequencePhoneBoundaries(m_withPhoneBoundaries ? utterance.size() : 0); // Phone boundaries of given sequence
        foreach_index(i, utterance)
        {
            if (m_withPhoneBoundaries)
                sequencePhoneBoundaries[i] = utterance[i].firstframe;
            const auto& timespan = utterance[i];
            if ((i == 0 && timespan.firstframe != 0) ||
                (i > 0 && utterance[i - 1].firstframe + utterance[i - 1].numframes != timespan.firstframe))
            {
                RuntimeError("Labels are not in the consecutive order MLF in label set: %ls", l.first.c_str());
            }

            if (timespan.classid >= dimension)
            {
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)timespan.classid, (int)dimension);
            }

            if (timespan.classid != static_cast<msra::dbn::CLASSIDTYPE>(timespan.classid))
            {
                RuntimeError("CLASSIDTYPE has too few bits");
            }

            if (SEQUENCELEN_MAX < timespan.firstframe + timespan.numframes)
            {
                RuntimeError("Maximum number of sample per sequence exceeded.");
            }

            numClasses = max(numClasses, (size_t)(1u + timespan.classid));

            for (size_t t = timespan.firstframe; t < timespan.firstframe + timespan.numframes; t++)
            {
                m_classIds.push_back(timespan.classid);
                numberOfFrames++;
            }
        }

        if (m_withPhoneBoundaries)
            m_phoneBoundaries.push_back(sequencePhoneBoundaries);

        description.m_numberOfSamples = numberOfFrames;
        m_utteranceIndex.push_back(totalFrames);
        totalFrames += numberOfFrames;

        if (m_keyToSequence.size() <= description.m_key.m_sequence)
        {
            m_keyToSequence.resize(description.m_key.m_sequence + 1, SIZE_MAX);
        }
        assert(m_keyToSequence[description.m_key.m_sequence] == SIZE_MAX);
        m_keyToSequence[description.m_key.m_sequence] = m_utteranceIndex.size() - 1;
        m_numberOfSequences++;
    }
    m_utteranceIndex.push_back(totalFrames);

    m_totalNumberOfFrames = totalFrames;

    fprintf(stderr, "MLFDataDeserializer::MLFDataDeserializer: %" PRIu64 " utterances with %" PRIu64 " frames in %" PRIu64 " classes\n",
            m_numberOfSequences,
            m_totalNumberOfFrames,
            numClasses);

    if (m_frameMode)
    {
        // Initializing array of labels.
        m_categories.reserve(dimension);
        m_categoryIndices.reserve(dimension);
        for (size_t i = 0; i < dimension; ++i)
        {
            auto category = make_shared<CategorySequenceData>();
            m_categoryIndices.push_back(static_cast<IndexType>(i));
            category->m_indices = &(m_categoryIndices[i]);
            category->m_nnzCounts.resize(1);
            category->m_nnzCounts[0] = 1;
            category->m_totalNnzCount = 1;
            category->m_numberOfSamples = 1;
            if (m_elementType == ElementType::tfloat)
            {
                category->m_data = &s_oneFloat;
            }
            else
            {
                assert(m_elementType == ElementType::tdouble);
                category->m_data = &s_oneDouble;
            }
            m_categories.push_back(category);
        }
    }
}

void MLFDataDeserializer::InitializeStream(const wstring& name, size_t dimension)
{
    // Initializing stream description - a single stream of MLF data.
    StreamDescriptionPtr stream = make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = name;
    stream->m_sampleLayout = make_shared<TensorShape>(dimension);
    stream->m_storageType = StorageType::sparse_csc;
    stream->m_elementType = m_elementType;
    m_streams.push_back(stream);
}

// Currently MLF has a single chunk.
// TODO: This will be changed when the deserializer properly supports chunking.
ChunkDescriptions MLFDataDeserializer::GetChunkDescriptions()
{
    auto cd = make_shared<ChunkDescription>();
    cd->m_id = 0;
    cd->m_numberOfSequences = m_frameMode ? m_totalNumberOfFrames : m_numberOfSequences;
    cd->m_numberOfSamples = m_totalNumberOfFrames;
    return ChunkDescriptions{cd};
}

// Gets sequences for a particular chunk.
void MLFDataDeserializer::GetSequencesForChunk(ChunkIdType, vector<SequenceDescription>& result)
{
    UNUSED(result);
    LogicError("Mlf deserializer does not support primary mode - it cannot control chunking.");
}

ChunkPtr MLFDataDeserializer::GetChunk(ChunkIdType chunkId)
{
    UNUSED(chunkId);
    assert(chunkId == 0);
    return make_shared<MLFChunk>(this);
};

// Sparse labels for an utterance.
template <class ElemType>
struct MLFSequenceData : SparseSequenceData
{
    vector<ElemType> m_values;
    unique_ptr<IndexType[]> m_indicesPtr;

    MLFSequenceData(size_t numberOfSamples) :
        m_values(numberOfSamples, 1),
        m_indicesPtr(new IndexType[numberOfSamples])
    {
        if (numberOfSamples > numeric_limits<IndexType>::max())
        {
            RuntimeError("Number of samples in an MLFSequence (%" PRIu64 ") "
                "exceeds the maximum allowed value (%" PRIu64 ")\n",
                numberOfSamples, (size_t)numeric_limits<IndexType>::max());
        }

        m_nnzCounts.resize(numberOfSamples, static_cast<IndexType>(1));
        m_numberOfSamples = (uint32_t) numberOfSamples;
        m_totalNnzCount = static_cast<IndexType>(numberOfSamples);
        m_indices = m_indicesPtr.get();
    }

    MLFSequenceData(size_t numberOfSamples, const vector<size_t>& phoneBoundaries) :
        MLFSequenceData(numberOfSamples)
    {
        for (auto boundary : phoneBoundaries)
            m_values[boundary] = PHONE_BOUNDARY;
    }

    const void* GetDataBuffer() override
    {
        return m_values.data();
    }
};

void MLFDataDeserializer::GetSequenceById(size_t sequenceId, vector<SequenceDataPtr>& result)
{
    if (m_frameMode)
    {
        size_t label = m_classIds[sequenceId];
        assert(label < m_categories.size());
        result.push_back(m_categories[label]);
    }
    else
    {
        // Packing labels for the utterance into sparse sequence.
        size_t startFrameIndex = m_utteranceIndex[sequenceId];
        size_t numberOfSamples = m_utteranceIndex[sequenceId + 1] - startFrameIndex;
        SparseSequenceDataPtr s;
        if (m_elementType == ElementType::tfloat)
        {
            if (m_withPhoneBoundaries)
                s = make_shared<MLFSequenceData<float>>(numberOfSamples, m_phoneBoundaries.at(sequenceId));
            else
                s = make_shared<MLFSequenceData<float>>(numberOfSamples);
        }
        else
        {
            assert(m_elementType == ElementType::tdouble);
            if (m_withPhoneBoundaries)
                s = make_shared<MLFSequenceData<double>>(numberOfSamples, m_phoneBoundaries.at(sequenceId));
            else
                s = make_shared<MLFSequenceData<double>>(numberOfSamples);
        }

        for (size_t i = 0; i < numberOfSamples; i++)
        {
            size_t frameIndex = startFrameIndex + i;
            size_t label = m_classIds[frameIndex];
            s->m_indices[i] = static_cast<IndexType>(label);
        }
        result.push_back(s);
    }
}

bool MLFDataDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
{

    auto sequenceId = key.m_sequence < m_keyToSequence.size() ? m_keyToSequence[key.m_sequence] : SIZE_MAX;

    if (sequenceId == SIZE_MAX)
    {
        return false;
    }

    result.m_chunkId = 0;
    result.m_key = key;

    if (m_frameMode)
    {
        size_t index = m_utteranceIndex[sequenceId] + key.m_sample;
        result.m_indexInChunk = index;
        result.m_numberOfSamples = 1;
    }
    else
    {
        assert(result.m_key.m_sample == 0);
        result.m_indexInChunk = sequenceId;
        result.m_numberOfSamples = (uint32_t) (m_utteranceIndex[sequenceId + 1] - m_utteranceIndex[sequenceId]);
    }
    return true;
}

}}}
