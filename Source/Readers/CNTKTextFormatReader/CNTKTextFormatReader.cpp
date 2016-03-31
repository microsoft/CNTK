//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKTextFormatReader.h"
#include "Config.h"
#include "TextConfigHelper.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "TextParser.h"

namespace Microsoft { namespace MSR { namespace CNTK {

CNTKTextFormatReader::CNTKTextFormatReader(MemoryProviderPtr provider,
    const ConfigParameters& config) :
    m_provider(provider)
{
    TextConfigHelper configHelper(config);

    try
    {
        if (configHelper.GetElementType() == ElementType::tfloat)
        {
            m_deserializer = shared_ptr<IDataDeserializer>(new TextParser<float>(configHelper));
        }
        else
        {
            m_deserializer = shared_ptr<IDataDeserializer>(new TextParser<double>(configHelper));
        }

        TransformerPtr randomizer;
        if (configHelper.ShouldRandomize())
        {
            randomizer = make_shared<BlockRandomizer>(0, SIZE_MAX, m_deserializer);
        }
        else
        {
            randomizer = std::make_shared<NoRandomizer>(m_deserializer);
        }

        randomizer->Initialize(nullptr, config);

        m_transformer = randomizer;
    }
    catch (const std::runtime_error& e)
    {
        RuntimeError("CNTKTextFormatReader: While reading '%ls': %s", configHelper.GetFilePath().c_str(), e.what());
    }
}

std::vector<StreamDescriptionPtr> CNTKTextFormatReader::GetStreamDescriptions()
{
    return m_deserializer->GetStreamDescriptions();
}

void CNTKTextFormatReader::StartEpoch(const EpochConfiguration& config)
{
    if (config.m_totalEpochSizeInSamples <= 0)
    {
        RuntimeError("Unsupported minibatch size '%d'.", (int)config.m_totalEpochSizeInSamples);
    }

    m_transformer->StartEpoch(config);
    m_packer = std::make_shared<SampleModePacker>(
        m_provider,
        m_transformer,
        config.m_minibatchSizeInSamples,
        GetStreamDescriptions());
}

Minibatch CNTKTextFormatReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}
} } }
