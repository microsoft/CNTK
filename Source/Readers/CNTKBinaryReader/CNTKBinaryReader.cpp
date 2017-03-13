//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKBinaryReader.h"
#include "Config.h"
#include "BinaryConfigHelper.h"
#include "BinaryChunkDeserializer.h"
#include "ChunkCache.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "SequencePacker.h"
#include "FramePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: This class should go away eventually.
// TODO: The composition of packer + randomizer + different deserializers in a generic manner is done in the CompositeDataReader.
// TODO: Currently preserving this for backward compatibility with current configs.
CNTKBinaryReader::CNTKBinaryReader(const ConfigParameters& config) 
{
    BinaryConfigHelper configHelper(config);

    std::stringstream log;
    log << "Initializing CNTKBinaryReader";
    try
    {
        m_deserializer = shared_ptr<IDataDeserializer>(new BinaryChunkDeserializer(configHelper));

        if (configHelper.ShouldKeepDataInMemory())
        {
            m_deserializer = shared_ptr<IDataDeserializer>(new ChunkCache(m_deserializer));
            log << " | keeping data in memory";
        }

        size_t window = configHelper.GetRandomizationWindow();
        if (window > 0)
        {
            // Verbosity is a general config parameter, not specific to the binary format reader.
            log << " | randomizing with window: " 
                << window 
                << configHelper.UseSampleBasedRandomizationWindow() ? " samples" : " chunks";
            int verbosity = config(L"verbosity", 0);
            m_sequenceEnumerator = make_shared<BlockRandomizer>(
                verbosity, /* verbosity */
                window,  /* randomizationRangeInSamples */
                m_deserializer, /* deserializer */
                true, /* shouldPrefetch */
                false, /* multithreadedGetNextSequences */
                 0, /*maxNumberOfInvalidSequences */
                configHelper.UseSampleBasedRandomizationWindow() /*sampleBasedRandomizationWindow */);
        }
        else
        {
            log << " | without randomization";
            m_sequenceEnumerator = std::make_shared<NoRandomizer>(m_deserializer);
        }

        m_packer = std::make_shared<SequencePacker>( m_sequenceEnumerator,
                                                    ReaderBase::GetStreamDescriptions());
    }
    catch (const std::runtime_error& e)
    {
        RuntimeError("CNTKBinaryReader: While reading '%ls': %s", configHelper.GetFilePath().c_str(), e.what());
    }
    if (configHelper.GetTraceLevel() > 2)
        fprintf(stderr, "%s\n", log.str().c_str());
}

} } }
