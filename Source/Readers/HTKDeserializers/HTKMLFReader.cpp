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
#include "FramePacker.h"
#include "SequencePacker.h"
#include "TruncatedBpttPacker.h"
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
    if (featureNames.size() < 1)
    {
        InvalidArgument("Network needs at least 1 feature specified.");
    }

    CorpusDescriptorPtr corpus = std::make_shared<CorpusDescriptor>();

    std::vector<IDataDeserializerPtr> featureDeserializers;
    std::vector<IDataDeserializerPtr> labelDeserializers;

    bool primary = true;
    // The first deserializer is the driving one, it defines chunking.
    // TODO: should we make this explicit configuration parameter
    for (const auto& featureName : featureNames)
    {
        auto deserializer = std::make_shared<HTKDataDeserializer>(corpus, readerConfig(featureName), featureName, primary);
        primary = false;
        featureDeserializers.push_back(deserializer);
    }

    for (const auto& labelName : labelNames)
    {
        auto deserializer = std::make_shared<MLFDataDeserializer>(corpus, readerConfig(labelName), labelName);

        labelDeserializers.push_back(deserializer);
    }

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
    if (deserializers.empty())
    {
        LogicError("Please specify at least a single input stream.");
    }

    bool cleanse = readerConfig(L"checkData", true);
    auto bundler = std::make_shared<Bundler>(readerConfig, deserializers[0], deserializers, cleanse);
    int verbosity = readerConfig(L"verbosity", 0);
    std::wstring readMethod = config.GetRandomizer();

    // TODO: this should be bool. Change when config per deserializer is allowed.
    if (AreEqualIgnoreCase(readMethod, std::wstring(L"blockRandomize")))
    {
        m_randomizer = std::make_shared<BlockRandomizer>(verbosity, window, bundler, true  /* should Prefetch */, BlockRandomizer::DecimationMode::chunk, true /* useLegacyRandomization */);
    }
    else if (AreEqualIgnoreCase(readMethod, std::wstring(L"none")))
    {
        m_randomizer = std::make_shared<NoRandomizer>(bundler);
    }
    else
    {
        RuntimeError("readMethod must be 'blockRandomize' or 'none'.");
    }

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
        m_packer = std::make_shared<FramePacker>(m_provider, m_randomizer, m_streams);
        break;
    case PackingMode::sequence:
        m_packer = std::make_shared<SequencePacker>(m_provider, m_randomizer, m_streams);
        break;
    case PackingMode::truncated:
        m_packer = std::make_shared<TruncatedBPTTPacker>(m_provider, m_randomizer, m_streams);
        break;
    default:
        LogicError("Unsupported type of packer '%d'.", (int)m_packingMode);
    }
}

std::vector<StreamDescriptionPtr> HTKMLFReader::GetStreamDescriptions()
{
    assert(!m_streams.empty());
    return m_streams;
}

void HTKMLFReader::StartEpoch(const EpochConfiguration& config)
{
    if (config.m_totalEpochSizeInSamples == 0)
    {
        RuntimeError("Epoch size cannot be 0.");
    }



    if (m_packingMode == PackingMode::truncated)
    {
        size_t minibatchSize = config.m_minibatchSizeInSamples;
        size_t truncationLength = m_truncationLength;
        if (truncationLength == 0)
        {
            // Old config, the truncation length is specified as the minibatch size.
            // In this case the truncation size is mbSize
            // and the real minibatch size is truncation size * nbruttsineachrecurrentiter
            fprintf(stderr, "Legacy configuration is used for truncated BPTT mode, please adapt the config to explicitly specify truncationLength.\n");
            truncationLength = minibatchSize;
            size_t numParallelSequences = m_numParallelSequencesForAllEpochs[config.m_epochIndex];
            minibatchSize = numParallelSequences * truncationLength;
        }
        
        EpochConfiguration bpttConfig;
        bpttConfig.m_numberOfWorkers = config.m_numberOfWorkers;
        bpttConfig.m_workerRank = config.m_workerRank;
        bpttConfig.m_totalEpochSizeInSamples = config.m_totalEpochSizeInSamples;
        bpttConfig.m_epochIndex = config.m_epochIndex;
        bpttConfig.m_minibatchSizeInSamples = minibatchSize;
        bpttConfig.m_truncationSize = truncationLength;

        m_randomizer->StartEpoch(bpttConfig);
        m_packer->StartEpoch(bpttConfig);
    }
    else
    {
        m_randomizer->StartEpoch(config);
        m_packer->StartEpoch(config);
    }
}

Minibatch HTKMLFReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}

}}}
