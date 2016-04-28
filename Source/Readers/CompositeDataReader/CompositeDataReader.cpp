//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CompositeDataReader.cpp : Defines a reader that allows composing different deserializers.
// With this reader in place the users should only extend deserializers.
//

#include "stdafx.h"
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#define DATAREADER_EXPORTS // creating the exports here

#include "CompositeDataReader.h"
#include "Bundler.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "FramePacker.h"
#include "SequencePacker.h"
#include "TruncatedBpttPacker.h"
#include "HeapMemoryProvider.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

CompositeDataReader::CompositeDataReader(const ConfigParameters& config, MemoryProviderPtr provider) : m_layout(make_shared<MBLayout>()),
    m_corpus(std::make_shared<CorpusDescriptor>()),
    m_provider(provider)
{
    // Identifying packing mode.
    bool frameMode = config(L"frameMode", true);
    bool truncated = config(L"truncated", false);
    if (frameMode && truncated)
    {
        LogicError("frameMode and truncated BPTT are mutually exclusive.");
    }

    if (frameMode)
    {
        m_packingMode = PackingMode::sample;
    }
    else if (truncated)
    {
        m_packingMode = PackingMode::truncated;
        m_truncationLength = config(L"truncationLength", 0);
        if (m_truncationLength == 0)
        {
            InvalidArgument("Truncation length cannot be 0.");
        }
    }
    else
    {
        m_packingMode = PackingMode::sequence;
    }

    // Whether we need to check data between different deserializers.
    bool cleanse = config(L"checkData", false);

    // Creating deserializers.
    // TODO: Currently the primary deserializer defines the corpus. The logic will be moved to CorpusDescriptor class.
    CreateDeserializers(config);

    // Bundling deserializers together.
    // TODO: Add transformers in between.
    auto bundler = std::make_shared<Bundler>(config, m_deserializers[0], m_deserializers, cleanse);

    int verbosity = config(L"verbosity", 2);

    // Pick up the randomizer.
    bool randomize = config(L"randomize", false);
    if (randomize)
    {
        // By default randomizing the whole data set.
        size_t randomizationWindow = config(L"randomizationWindow", requestDataSize);
        m_randomizer = std::make_shared<BlockRandomizer>(verbosity, randomizationWindow, bundler, BlockRandomizer::DecimationMode::chunk, true);
    }
    else
    {
        m_randomizer = std::make_shared<NoRandomizer>(bundler);
    }

    m_randomizer->Initialize(nullptr, config);

    // Create output stream descriptions - where to get those? from config? what if it is not the same as network expects?
    // TODO: Currently only sparse streams.
    for (const auto& streamDescription : bundler->GetStreamDescriptions())
    {
        StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*streamDescription);
        stream->m_storageType = StorageType::dense;
        m_streams.push_back(stream);
        m_nameToStreamId.insert(std::make_pair(streamDescription->m_name, streamDescription->m_id));
    }
}

std::vector<StreamDescriptionPtr> CompositeDataReader::GetStreamDescriptions()
{
    return m_streams;
}

Minibatch CompositeDataReader::ReadMinibatch()
{
    return m_packer->ReadMinibatch();
}

void CompositeDataReader::CreateDeserializers(const ConfigParameters& readerConfig)
{
    argvector<ConfigValue> deserializerConfigs =
        readerConfig(L"deserializers", ConfigParameters::Array(argvector<ConfigValue>(vector<ConfigValue> {})));

    assert(m_deserializers.empty());
    bool primary = true;  // CUrrently, the first deserializer becomes primary - it drives chunking.
    for (size_t i = 0; i < deserializerConfigs.size(); ++i)
    {
        // TODO: Should go away in the future. Framing can be done on top of deserializers.
        ConfigParameters p = deserializerConfigs[i];
        p.Insert("frameMode", m_packingMode == PackingMode::sample ? "true" : "false");
        p.Insert("precision", m_precision);

        IDataDeserializerPtr d = CreateDeserializer(p, primary);
        primary = false;
        m_deserializers.push_back(d);
    }
}

IDataDeserializerPtr CompositeDataReader::CreateDeserializer(const ConfigParameters& deserializerConfig, bool primary)
{
    typedef bool(*CreateDeserializerFactory) (IDataDeserializer** d, const std::wstring& type, const ConfigParameters& cfg, CorpusDescriptorPtr corpus, bool primary);

    std::string deserializerModule = deserializerConfig("module");
    CreateDeserializerFactory f = (CreateDeserializerFactory)Plugin::Load(deserializerModule, "CreateDeserializer");

    std::wstring deserializerType = deserializerConfig("type");
    IDataDeserializer* d;
    if (!f(&d, deserializerType, deserializerConfig, m_corpus, primary))
    {
        RuntimeError("Cannot create deserializer. Please check module and type in the configuration.");
    }

    assert(d != nullptr);
    return IDataDeserializerPtr(d);
}

void CompositeDataReader::StartEpoch(const EpochConfiguration& cfg)
{
    EpochConfiguration config = cfg;

    if (config.m_totalEpochSizeInSamples <= 0)
    {
        RuntimeError("Unsupported minibatch size '%d'.", (int)config.m_totalEpochSizeInSamples);
    }

    m_randomizer->StartEpoch(config);

    // TODO: As the next step the packers should be moved into the network.
    switch (m_packingMode)
    {
    case PackingMode::sample:
        m_packer = std::make_shared<FramePacker>(
            m_provider,
            m_randomizer,
            m_streams);
        break;
    case PackingMode::sequence:
        m_packer = std::make_shared<SequencePacker>(
            m_provider,
            m_randomizer,
            m_streams);
        break;
    case PackingMode::truncated:
    {
        config.m_truncationSize = m_truncationLength;
        m_packer = std::make_shared<TruncatedBPTTPacker>(
            m_provider,
            m_randomizer,
            m_streams);
        break;
    }
    default:
        LogicError("Unsupported type of packer '%d'.", (int)m_packingMode);
    }

    m_packer->StartEpoch(config);
}

}}}