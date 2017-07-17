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
#include "ReaderConstants.h"
#include "V2Dependencies.h"
#include "LTNoRandomizer.h"
#include "LTTumblingWindowRandomizer.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

// The whole CompositeDataReader is meant as a stopgap to allow deserializers/transformers composition until SGD talkes 
// directly to the new Reader API. 
// For more information please see its header file.
// This method composes together packers + randomizer + a set of transformers and deserializers.
CompositeDataReader::CompositeDataReader(const ConfigParameters& config) :
    m_truncationLength(0)
{
    wstring action = config(L"action", L"");
    bool isActionWrite = AreEqualIgnoreCase(action, L"write");

    // By default, we use numeric sequence keys (i.e., for cbf, ctf, image and base64 readers).
    // For MLF and HTK deserializers, we use non-numeric (string) sequence keys.
    bool useNumericSequenceKeys = true;
    if (ContainsDeserializer(config, L"HTKFeatureDeserializer") ||
        ContainsDeserializer(config, L"HTKMLFDeserializer")) 
    {
        useNumericSequenceKeys = false;
    }

    useNumericSequenceKeys = config(L"useNumericSequenceKeys", useNumericSequenceKeys);

    bool useHash = config(L"hashSequenceKeys", false);
    m_corpus = std::make_shared<CorpusDescriptor>(useNumericSequenceKeys, useHash);

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
    bool composable = CreateDeserializers(config);
    if (m_deserializers.empty())
        InvalidArgument("Could not find deserializers in the reader config.");

    if (!composable && m_deserializers.size() > 1)
        InvalidArgument("Currently user defined deserializers do not support composability. Please specify a single deserializer.");

    DataDeserializerPtr deserializer = m_deserializers.front();
    if (m_deserializers.size() > 1)
    {
        // Bundling deserializers together.
        // Option whether we need to check data between different deserializers.
        bool cleanse = config(L"checkData", true);
        deserializer = std::make_shared<Bundler>(config, deserializer, m_deserializers, cleanse);
    }

    int verbosity = config(L"verbosity", 0);

    // Pick up the randomizer, always picking up no randomization for the write mode.
    bool randomize = isActionWrite ? false : config(L"randomize", true);

    // Get maximum number of allowed errors per worker.
    size_t maxErrors = config(L"maxErrors", 0);

    // By default do not use omp threads for deserialization of sequences.
    // It makes sense to put it to true for cases when deserialization is CPU intensive,
    // i.e. decompression of images.
    bool multiThreadedDeserialization = config(L"multiThreadedDeserialization", ContainsDeserializer(config, L"ImageDeserializer"));

    if (!composable) // Pick up simple interface.
    {
        if (randomize)
        {
            bool sampleBasedRandomizationWindow = config(L"sampleBasedRandomizationWindow", false);
            m_sequenceEnumerator = std::make_shared<LTTumblingWindowRandomizer>(deserializer,
                sampleBasedRandomizationWindow, config(L"randomizationWindow", requestDataSize),
                GetRandomSeed(config),
                multiThreadedDeserialization, maxErrors);
        }
        else
            m_sequenceEnumerator = std::make_shared<LTNoRandomizer>(deserializer, multiThreadedDeserialization, maxErrors);
    }
    else
    {
        if (randomize)
        {
            // By default randomizing the whole data set.
            size_t randomizationWindow = requestDataSize;

            // Currently in case of images, a single chunk is a single image. So no need to randomize, chunks will be randomized anyway.
            if (ContainsDeserializer(config, L"ImageDeserializer") && m_deserializers.size() == 1)
            {
                randomizationWindow = 1;
                m_packingMode = PackingMode::sample;
            }

            randomizationWindow = config(L"randomizationWindow", randomizationWindow);
            bool sampleBasedRandomizationWindow = config(L"sampleBasedRandomizationWindow", true);

            if (ContainsDeserializer(config, L"CNTKTextFormatDeserializer") && !config.ExistsCurrent(L"randomizationWindow"))
            {
                if (!config.ExistsCurrent(L"sampleBasedRandomizationWindow") || // sampleBasedRandomizationWindow is not specified
                    !sampleBasedRandomizationWindow) // randomization window is in chunks
                {
                    sampleBasedRandomizationWindow = false;
                    size_t chunkSizeBytes = config(L"chunkSizeInBytes", g_32MB); // 32 MB by default
                    randomizationWindow = g_4GB / chunkSizeBytes; // ~ 4 GB disk space worth of chunks
                                                                  // TODO: decrease randomization window if m_deserializers.size() > 1 ?
                }
                else
                {
                    // config explicitly says to use a sample-based window, but does not specify its size.
                    LogicError("'sampleBasedRandomizationWindow' (== 'true') requires that the 'randomizationWindow' is explicitly specified.");
                }
            }

            bool shouldPrefetch = true;
            m_sequenceEnumerator = std::make_shared<BlockRandomizer>(verbosity, randomizationWindow, deserializer, shouldPrefetch,
                multiThreadedDeserialization, maxErrors, sampleBasedRandomizationWindow, GetRandomSeed(config));
        }
        else
            m_sequenceEnumerator = std::make_shared<NoRandomizer>(deserializer, multiThreadedDeserialization, maxErrors);
    }

    // In case when there are transforms, applying them to the data.
    m_sequenceEnumerator = m_transforms.empty()
        ? m_sequenceEnumerator
        : std::make_shared<TransformController>(m_transforms, m_sequenceEnumerator);

    // TODO: Output stream descriptions - this should come from the network so that we can check 
    // that input matches what the network expects (including tensor shape, etc.).
    std::vector<StreamInformation> outputStreams = m_sequenceEnumerator->GetStreamDescriptions();

    // Currently for prefetch we use two alternating buffers,
    // same is the default.
    size_t numAlternatingBuffers = 2;

    // Check whether to use local timeline, by default we use it for better performance.
    bool localTimeline = config(L"localTimeline", true);
    switch (m_packingMode)
    {
    case PackingMode::sample:
        m_packer = std::make_shared<FramePacker>(
            m_sequenceEnumerator,
            outputStreams,
            numAlternatingBuffers,
            localTimeline,
            m_corpus);
        break;
    case PackingMode::sequence:
        m_packer = std::make_shared<SequencePacker>(
            m_sequenceEnumerator,
            outputStreams,
            numAlternatingBuffers,
            localTimeline,
            m_corpus);
        break;
    case PackingMode::truncated:
    {
        // Currently BPTT does not support sparse format as output.
        // We always require dense from the packer.
        for (auto& s : outputStreams)
            s.m_storageFormat = StorageFormat::Dense;

        m_packer = std::make_shared<TruncatedBPTTPacker>(
            m_sequenceEnumerator,
            outputStreams,
            numAlternatingBuffers,
            m_corpus);
        break;
    }
    default:
        LogicError("Unsupported type of packer '%d'.", (int)m_packingMode);
    }
}

std::vector<StreamInformation> CompositeDataReader::GetStreamDescriptions()
{
    return m_packer->GetStreamDescriptions();
}

// Create deserializers based on the specified configuration. 
// deserializers = [
//        [ type = "ImageDataDeserializer" module = "ImageReader" ...]
//        [ type = "CNTKTextFormatDeserializer" module = "CNTKTextFormatReader" ...]
bool CompositeDataReader::CreateDeserializers(const ConfigParameters& readerConfig)
{
    argvector<ConfigValue> deserializerConfigs =
        readerConfig(L"deserializers", ConfigParameters::Array(argvector<ConfigValue>(vector<ConfigValue> {})));

    assert(m_deserializers.empty());

    auto traceLevel = readerConfig.Find("traceLevel");
    bool composable = true;

    bool primary = true;  // Currently, the first deserializer becomes primary - it drives chunking.
    for (size_t i = 0; i < deserializerConfigs.size(); ++i)
    {
        // TODO: Should go away in the future. Framing can be done on top of deserializers.
        ConfigParameters p = deserializerConfigs[i];
        p.Insert("frameMode", m_packingMode == PackingMode::sample ? "true" : "false");
        p.Insert("precision", m_precision);
        if (!traceLevel.empty()) 
        {
            p.Insert("traceLevel", traceLevel);
        }

        composable &= p(L"composable", true);
        DataDeserializerPtr d = CreateDeserializer(p, primary);
        primary = false;
        m_deserializers.push_back(d);
    }
    return composable;
}

// Creates a particular deserializer based on the config: its loads the external module and calls CreateDeserializer
// factory function for a particular deserializer type.
DataDeserializerPtr CompositeDataReader::CreateDeserializer(const ConfigParameters& deserializerConfig, bool primary)
{
    typedef bool(*CreateDeserializerFactory) (DataDeserializerPtr& d, const std::wstring& type, const ConfigParameters& cfg, CorpusDescriptorPtr corpus, bool primary);

    std::string deserializerModule = deserializerConfig("module");
    CreateDeserializerFactory f = (CreateDeserializerFactory)Plugin::Load(deserializerModule, "CreateDeserializer");

    std::wstring deserializerType = deserializerConfig("type");
    DataDeserializerPtr d;
    if (!f(d, deserializerType, deserializerConfig, m_corpus, primary))
    {
        RuntimeError("Cannot create deserializer. Please check module and type in the configuration.");
    }

    // Create transformers if necessary.
    CreateTransforms(deserializerConfig);

    assert(d != nullptr);
    return d;
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
    if (!deserializerConfig.Exists("input"))
        return;

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

        // Read transformers in order and appending them to the transformer pipeline.
        argvector<ConfigParameters> transforms = input("transforms");
        for (size_t j = 0; j < transforms.size(); ++j)
        {
            ConfigParameters p = transforms[j];
            p.Insert("precision", deserializerConfig("precision"));

            TransformerPtr transformer = CreateTransformer(p, defaultModule, std::wstring());
            m_transforms.push_back(Transformation{transformer, inputName});
        }

        // Let's add a cast transformer by default. It is noop if the type provided by others is float
        // or double, but will do a proper cast if the type is uchar.
        auto cast = CreateTransformer(input, defaultModule, std::wstring(L"Cast"));
        m_transforms.push_back(Transformation{ cast, inputName });
    }
}

// Create a transformer for a particular configuration. Loading it from the module of the deserializer if module is not specified, i.e.
//     transforms = [
//         [type = "Scale" width=...]:...
TransformerPtr CompositeDataReader::CreateTransformer(const ConfigParameters& config, const string& defaultModule, const std::wstring& type)
{
    typedef bool(*TransformerFactory) (Transformer** t, const std::wstring& type, const ConfigParameters& cfg);

    std::string transformerModule = config("module", defaultModule.c_str());
    TransformerFactory f = (TransformerFactory)Plugin::Load(transformerModule, "CreateTransformer");

    std::wstring transformerType = type.empty() ? config("type") : type;
    Transformer* t;
    if (!f(&t, transformerType, config))
    {
        RuntimeError("Cannot create transformer. Please check the module and type in the configuration.");
    }

    assert(t != nullptr);
    return TransformerPtr(t);
}

void CompositeDataReader::StartEpoch(const EpochConfiguration& cfg, const std::map<std::wstring, int>& inputDescriptions)
{
    EpochConfiguration config = cfg;
    if (m_packingMode == PackingMode::truncated)
    {
        config.m_truncationSize = m_truncationLength;
    }

    ReaderBase::StartEpoch(config, inputDescriptions);
}

bool CompositeDataReader::ContainsDeserializer(const ConfigParameters& readerConfig, const wstring& type)
{
    argvector<ConfigValue> deserializerConfigs =
        readerConfig(L"deserializers", ConfigParameters::Array(argvector<ConfigValue>(vector<ConfigValue> {})));

    for (size_t i = 0; i < deserializerConfigs.size(); ++i)
    {
        ConfigParameters p = deserializerConfigs[i];
        std::wstring deserializerType = p("type");
        if (deserializerType == type)
            return true;
    }
    return false;
}

}
