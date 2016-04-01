//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "HTKMLFReader.h"
#include "Config.h"
#include "HTKDataDeserializer.h"
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "Bundler.h"
#include "StringUtil.h"
#include "SequencePacker.h"
#include "SampleModePacker.h"
#include "BpttPacker.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

std::vector<IDataDeserializerPtr> CreateDeserializers(const ConfigParameters& readerConfig)
{
    std::vector<std::wstring> featureNames;
    std::vector<std::wstring> labelNames;
    std::vector<std::wstring> notused;
    ConfigHelper config(readerConfig);

    config.GetDataNamesFromConfig(featureNames, labelNames, notused, notused);
    if (featureNames.size() < 1 || labelNames.size() < 1)
    {
        InvalidArgument("Network needs at least 1 feature and 1 label specified.");
    }

    CorpusDescriptorPtr corpus = std::make_shared<CorpusDescriptor>();

    std::vector<IDataDeserializerPtr> featureDeserializers;
    std::vector<IDataDeserializerPtr> labelDeserializers;

    for (const auto& featureName : featureNames)
    {
        auto deserializer = std::make_shared<HTKDataDeserializer>(corpus, readerConfig(featureName), featureName);
        featureDeserializers.push_back(deserializer);
    }
    assert(featureDeserializers.size() == 1);

    for (const auto& labelName : labelNames)
    {
        auto deserializer = std::make_shared<MLFDataDeserializer>(corpus, readerConfig(labelName), labelName);

        labelDeserializers.push_back(deserializer);
    }
    assert(labelDeserializers.size() == 1);

    std::vector<IDataDeserializerPtr> deserializers;
    deserializers.insert(deserializers.end(), featureDeserializers.begin(), featureDeserializers.end());
    deserializers.insert(deserializers.end(), labelDeserializers.begin(), labelDeserializers.end());

    return deserializers;
}

HTKMLFReader::HTKMLFReader(MemoryProviderPtr provider,
    const ConfigParameters& readerConfig)
    : m_seed(0), m_provider(provider)
{
    // TODO: deserializers and transformers will be dynamically loaded
    // from external libraries based on the configuration/brain script.

    bool frameMode = readerConfig(L"frameMode", true);
    bool truncated = readerConfig(L"truncated", false);
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
    }
    else
    {
        m_packingMode = PackingMode::sequence;
    }

    // nbruttsineachrecurrentiter is old reader configuration, truncationLength is the new one.
    // If truncation length is specified we estimate
    // the number of parallel sequences we have to pack as max(1, (mbsize/truncationLength))
    // If nbruttsineachrecurrentiter is specified we assume that the truncation size is mbSize
    // and the real minibatch size in mbSize * nbruttsineachrecurrentiter[epochIndex]
    m_truncationLength = readerConfig(L"truncationLength", 0);
    m_numParallelSequencesForAllEpochs =
        readerConfig(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int> { 1 })));

    ConfigHelper config(readerConfig);
    size_t window = config.GetRandomizationWindow();
    auto deserializers = CreateDeserializers(readerConfig);
    assert(deserializers.size() == 2);

    auto bundler = std::make_shared<Bundler>(readerConfig, deserializers[0], deserializers, false);
    int verbosity = readerConfig(L"verbosity", 2);
    std::wstring readMethod = config.GetRandomizer();

    // TODO: this should be bool. Change when config per deserializer is allowed.
    if (AreEqualIgnoreCase(readMethod, std::wstring(L"blockRandomize")))
    {
        m_randomizer = std::make_shared<BlockRandomizer>(verbosity, window, bundler, BlockRandomizer::DecimationMode::chunk, true /* useLegacyRandomization */);
    }
    else if (AreEqualIgnoreCase(readMethod, std::wstring(L"none")))
    {
        m_randomizer = std::make_shared<NoRandomizer>(bundler);
    }
    else
    {
        RuntimeError("readMethod must be 'blockRandomize' or 'none'.");
    }

    m_randomizer->Initialize(nullptr, readerConfig);

    // Create output stream descriptions (all dense)
    for (auto d : deserializers)
    {
        for (auto i : d->GetStreamDescriptions())
        {
            StreamDescriptionPtr stream = std::make_shared<StreamDescription>(*i);
            stream->m_storageType = StorageType::dense;
            stream->m_id = m_streams.size();
            m_streams.push_back(stream);
        }
    }
}

std::vector<StreamDescriptionPtr> HTKMLFReader::GetStreamDescriptions()
{
    assert(!m_streams.empty());
    return m_streams;
}

void HTKMLFReader::StartEpoch(const EpochConfiguration& config)
{
    if (config.m_totalEpochSizeInSamples <= 0)
    {
        RuntimeError("Unsupported minibatch size '%d'.", (int)config.m_totalEpochSizeInSamples);
    }

    m_randomizer->StartEpoch(config);

    // TODO: should we unify sample and sequence mode packers into a single one.
    // TODO: functionally they are the same, the only difference is how we handle
    // TODO: MBlayout and what is the perf hit for iterating/copying sequences.
    // TODO: Should do more perf tests before unifying these two.

    // TODO: As the next step the packers will be moved out of the readers into the
    // TODO: core CNTK. They are format agnostic and can be used with any type of 
    // TODO: deserializers.
    switch (m_packingMode)
    {
    case PackingMode::sample:
        m_packer = std::make_shared<SampleModePacker>(
            m_provider,
            m_randomizer,
            config.m_minibatchSizeInSamples,
            m_streams);
        break;
    case PackingMode::sequence:
        m_packer = std::make_shared<SequencePacker>(
            m_provider,
            m_randomizer,
            config.m_minibatchSizeInSamples,
            m_streams);
        break;
    case PackingMode::truncated:
    {
        size_t minibatchSize = config.m_minibatchSizeInSamples;
        size_t truncationLength = m_truncationLength;
        if (truncationLength == 0)
        {
            // Old config, the truncation length is specified as the minibatch size.
            // In this case the truncation size is mbSize
            // and the real minibatch size is truncation size * nbruttsineachrecurrentiter
            fprintf(stderr, "Legacy configuration is used for truncated BPTT mode, please adapt the config to explicitly specify truncationLength.");
            truncationLength = minibatchSize;
            size_t numParallelSequences = m_numParallelSequencesForAllEpochs[config.m_epochIndex];
            minibatchSize = numParallelSequences * truncationLength;
        }

        m_packer = std::make_shared<BpttPacker>(
            m_provider,
            m_randomizer,
            minibatchSize,
            truncationLength,
            m_streams);
        break;
    }
    default:
        LogicError("Unsupported type of packer '%d'.", (int)m_packingMode);
    }
}

Minibatch HTKMLFReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}

}}}
