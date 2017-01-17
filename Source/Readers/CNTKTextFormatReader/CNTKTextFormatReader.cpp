//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKTextFormatReader.h"
#include "Config.h"
#include "TextConfigHelper.h"
#include "ChunkCache.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "TextParser.h"
#include "SequencePacker.h"
#include "FramePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: This class should go away eventually.
// TODO: The composition of packer + randomizer + different deserializers in a generic manner is done in the CompositeDataReader.
// TODO: Currently preserving this for backward compatibility with current configs.
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

        if (configHelper.ShouldKeepDataInMemory()) 
        {
            m_deserializer = shared_ptr<IDataDeserializer>(new ChunkCache(m_deserializer));
        }

        size_t window = configHelper.GetRandomizationWindow();
        if (window > 0)
        {
            // Verbosity is a general config parameter, not specific to the text format reader.
            int verbosity = config(L"verbosity", 0);
            m_randomizer = make_shared<BlockRandomizer>(verbosity, window, m_deserializer, true);
        }
        else
        {
            m_randomizer = std::make_shared<NoRandomizer>(m_deserializer);
        }

        if (configHelper.IsInFrameMode()) 
        {
            m_packer = std::make_shared<FramePacker>(
                m_provider,
                m_randomizer,
                GetStreamDescriptions());
        }
        else
        {
        m_packer = std::make_shared<SequencePacker>(
            m_provider,
            m_randomizer,
            GetStreamDescriptions());
        }
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
    if (config.m_totalEpochSizeInSamples == 0)
    {
        RuntimeError("Epoch size cannot be 0.");
    }

    m_randomizer->StartEpoch(config);
    m_packer->StartEpoch(config);
}

Minibatch CNTKTextFormatReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}
} } }
