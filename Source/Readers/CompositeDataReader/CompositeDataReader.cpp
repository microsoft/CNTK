//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CompositeReader.cpp : Defines a reader that allows composing different deserializers.
// With this reader in place the users should only extend deserializers.
//

#include "stdafx.h"
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#include "CompositeDataReader.h"
#include "Bundler.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "FramePacker.h"
#include "SequencePacker.h"
#include "TruncatedBpttPacker.h"
#include "CorpusDescriptor.h"
#include "ConfigUtil.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The whole CompositeDataReader is meant as a stopgap to allow deserializers/transformers composition until SGD talkes 
// directly to the new Reader API. 
// For more information please see its header file.
// This method composes together packers + randomizer + a set of transformers and deserializers.
CompositeDataReader::CompositeDataReader(const ConfigParameters& config, MemoryProviderPtr provider) : m_layout(make_shared<MBLayout>()),
    m_corpus(std::make_shared<CorpusDescriptor>()),
    m_provider(provider)
{
    wstring action = config(L"action", L"");
    bool isActionWrite = AreEqualIgnoreCase(action, L"write");

    // Identifying packing mode.
    bool frameMode = config(L"frameMode", false);
    bool truncated = config(L"truncated", false);
    if (frameMode && truncated)
    {
        LogicError("frameMode and truncated BPTT are mutually exclusive.");
    }

    if (isActionWrite) // For writing we always use sequence mode.
    {
        m_packingMode = PackingMode::sequence;
    }
    else if (frameMode)
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

    m_precision = config("precision", "float");

    // Creating deserializers.
    // TODO: Currently the primary deserializer defines the corpus. The logic will be moved to CorpusDescriptor class.
    CreateDeserializers(config);

    if (m_deserializers.empty())
    {
        InvalidArgument("Could not find deserializers in the reader config.");
    }

    IDataDeserializerPtr deserializer = m_deserializers.front();
    if (m_deserializers.size() > 1)
    {
        // Bundling deserializers together.
        // Option whether we need to check data between different deserializers.
        bool cleanse = config(L"checkData", true);
        deserializer = std::make_shared<Bundler>(config, deserializer, m_deserializers, cleanse);
    }

    int verbosity = config(L"verbosity", 0);

    // Pick up the randomizer, always picking up no randomization for the write mode.
    bool randomize = isActionWrite ? false : config(L"randomize", false);

    // By default do not use omp threads for deserialization of sequences.
    // It makes sense to put it to true for cases when deserialization is CPU intensive,
    // i.e. decompression of images.
    bool multiThreadedDeserialization = config(L"multiThreadedDeserialization", false);
    if (randomize)
    {
        // By default randomizing the whole data set.
        size_t randomizationWindow = config(L"randomizationWindow", requestDataSize);
        // By default using STL random number generator.
        bool useLegacyRandomization = config(L"useLegacyRandomization", false);
        m_sequenceEnumerator = std::make_shared<BlockRandomizer>(verbosity, randomizationWindow, deserializer, true, BlockRandomizer::DecimationMode::chunk, useLegacyRandomization, multiThreadedDeserialization);
    }
    else
    {
        m_sequenceEnumerator = std::make_shared<NoRandomizer>(deserializer, multiThreadedDeserialization);
    }

    // In case when there are transforms, applying them to the data.
    m_sequenceEnumerator = m_transforms.empty()
        ? m_sequenceEnumerator 
        : std::make_shared<TransformController>(m_transforms, m_sequenceEnumerator);

    // Create output stream descriptions - where to get those? from config? what if it is not the same as network expects?
    // TODO: Currently only dense output streams.
    // TODO: Check here. We should already support repacking sparse into dense in the shim/matrix.
    for (const auto& streamDescription : m_sequenceEnumerator->GetStreamDescriptions())
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

// Create deserializers based on the specified configuration. 
// deserializers = [
//        [ type = "ImageDataDeserializer" module = "ImageReader" ...]
//        [ type = "CNTKTextFormatDeserializer" module = "CNTKTextFormatReader" ...]
void CompositeDataReader::CreateDeserializers(const ConfigParameters& readerConfig)
{
    argvector<ConfigValue> deserializerConfigs =
        readerConfig(L"deserializers", ConfigParameters::Array(argvector<ConfigValue>(vector<ConfigValue> {})));

    assert(m_deserializers.empty());
    bool primary = true;  // Currently, the first deserializer becomes primary - it drives chunking.
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

// Creates a particular deserializer based on the config: its loads the external module and calls CreateDeserializer
// factory function for a particular deserializer type.
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

    // Create transformers if necessary.
    CreateTransforms(deserializerConfig);

    assert(d != nullptr);
    return IDataDeserializerPtr(d);
}

// Create transformers based on the configuration, i.e.
// deserializers = [
//     [
//         type = "ImageDataDeserializer"
//         module = "ImageReader"
//         inputs = [
//               features = [
//---->              transforms = [
//                       [type = "Crop"]:[type = "Scale"]...

void CompositeDataReader::CreateTransforms(const ConfigParameters& deserializerConfig)
{
    std::string defaultModule = deserializerConfig("module");
    argvector<ConfigParameters> inputs = deserializerConfig("input");
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        // Trying to find transfomers in a stream section of the config.
        auto inputSections = TryGetSectionsWithParameter(inputs[i], "transforms");
        if (inputSections.size() > 1)
        {
            LogicError("Only a single 'transforms' config is allowed per stream.");
        }

        // No need to create anything for this stream, skipping.
        if (inputSections.empty())
        {
            continue;
        }

        ConfigParameters input = inputs[i](inputSections.front());
        std::wstring inputName = msra::strfun::utf16(input.ConfigName());

        // Read tranformers in order and appending them to the transformer pipeline.
        argvector<ConfigParameters> transforms = input("transforms");
        for (size_t j = 0; j < transforms.size(); ++j)
        {
            TransformerPtr transformer = CreateTransformer(transforms[j], defaultModule);
            m_transforms.push_back(Transformation{transformer, inputName});
        }
    }

}

// Create a transformer for a particular configuration. Loading it from the module of the deserializer if module is not specified, i.e.
//     transforms = [
//         [type = "Scale" width=...]:...
TransformerPtr CompositeDataReader::CreateTransformer(const ConfigParameters& config, const string& defaultModule)
{
    typedef bool(*TransformerFactory) (Transformer** t, const std::wstring& type, const ConfigParameters& cfg);

    std::string transformerModule = config("module", defaultModule.c_str());
    TransformerFactory f = (TransformerFactory)Plugin::Load(transformerModule, "CreateTransformer");

    std::wstring transformerType = config("type");
    Transformer* t;
    if (!f(&t, transformerType, config))
    {
        RuntimeError("Cannot create transformer. Please check the module and type in the configuration.");
    }

    assert(t != nullptr);
    return TransformerPtr(t);
}

void CompositeDataReader::StartEpoch(const EpochConfiguration& cfg)
{
    EpochConfiguration config = cfg;

    if (config.m_totalEpochSizeInSamples <= 0)
    {
        RuntimeError("Unsupported epoch size '%d'.", (int)config.m_totalEpochSizeInSamples);
    }

    m_sequenceEnumerator->StartEpoch(config);

    // TODO: As the next step the packers should be moved into the network.
    switch (m_packingMode)
    {
    case PackingMode::sample:
        m_packer = std::make_shared<FramePacker>(
            m_provider,
            m_sequenceEnumerator,
            m_streams);
        break;
    case PackingMode::sequence:
        m_packer = std::make_shared<SequencePacker>(
            m_provider,
            m_sequenceEnumerator,
            m_streams);
        break;
    case PackingMode::truncated:
    {
        config.m_truncationSize = m_truncationLength;
        m_packer = std::make_shared<TruncatedBPTTPacker>(
            m_provider,
            m_sequenceEnumerator,
            m_streams);
        break;
    }
    default:
        LogicError("Unsupported type of packer '%d'.", (int)m_packingMode);
    }

    m_packer->StartEpoch(config);
}

}}}