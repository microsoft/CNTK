//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "TextReader.h"
#include "Config.h"
#include "TextConfigHelper.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "TextParser.h"
#include <omp.h>

namespace Microsoft { namespace MSR { namespace CNTK {

TextReader::TextReader(MemoryProviderPtr provider,
                         const ConfigParameters& config)
    : m_seed(0), m_provider(provider)
{
    // In the future, deserializers and transformers will be dynamically loaded
    // from external libraries based on the configuration/brain script.
    // We will provide ability to implement the transformer and
    // deserializer interface not only in C++ but in scripting languages as well.

    TextConfigHelper configHelper(config);

    int threadCount = configHelper.GetCpuThreadCount();
    if (threadCount > 0)
    {
        omp_set_num_threads(threadCount);
    }

    m_parser = std::make_shared<TextParser>(configHelper.GetFilepath(), configHelper.GetInputStreams());


    for (const StreamDescriptor& stream : configHelper.GetOutputStreams())
    {
        auto streamDescription = std::make_shared<StreamDescription>(stream);
        streamDescription->m_sampleLayout = std::make_shared<TensorShape>(stream.m_sampleSize);
        m_streams.push_back(streamDescription);
    }

    TransformerPtr randomizer;
    if (configHelper.ShouldRandomize())
    {
        randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, m_parser);
    }
    else
    {
        randomizer = std::make_shared<NoRandomizer>(m_parser);
    }

    randomizer->Initialize(nullptr, config);

    m_transformer = randomizer;
}

std::vector<StreamDescriptionPtr> TextReader::GetStreamDescriptions()
{
    return m_streams;
}

void TextReader::StartEpoch(const EpochConfiguration& config)
{
    if (config.m_totalEpochSizeInSamples <= 0)
    {
        RuntimeError("Unsupported minibatch size '%u'.", (int)config.m_totalEpochSizeInSamples);
    }

    m_transformer->StartEpoch(config);
    m_packer = std::make_shared<SampleModePacker>(
        m_provider,
        m_transformer,
        config.m_minibatchSizeInSamples,
        m_streams);
}

Minibatch TextReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}
} } }
