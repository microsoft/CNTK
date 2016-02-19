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

    std::vector<std::wstring> sequences = ConfigHelper(readerConfig(featureNames.front())).GetFeaturePaths();
    CorpusDescriptorPtr corpus = std::make_shared<CorpusDescriptor>(std::move(sequences));

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

    assert(readerConfig(L"frameMode", true));
    ConfigHelper config(readerConfig);

    size_t window = config.GetRandomizationWindow();
    auto deserializers = CreateDeserializers(readerConfig);
    assert(deserializers.size() == 2);

    auto bundler = std::make_shared<Bundler>(readerConfig, deserializers[0], deserializers);

    std::wstring readMethod = config.GetRandomizer();
    if (!AreEqualIgnoreCase(readMethod, std::wstring(L"blockRandomize")))
    {
        RuntimeError("readMethod must be 'blockRandomize'");
    }

    int verbosity = readerConfig(L"verbosity", 2);
    m_randomizer = std::make_shared<BlockRandomizer>(verbosity, window, bundler, BlockRandomizer::DistributionMode::chunk_modulus, true /* useLegacyRandomization */);
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
    m_packer = std::make_shared<SampleModePacker>(
        m_provider,
        m_randomizer,
        config.m_minibatchSizeInSamples,
        m_streams);
}

Minibatch HTKMLFReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}

}}}
