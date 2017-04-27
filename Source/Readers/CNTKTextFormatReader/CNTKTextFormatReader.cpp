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

using namespace std;

// TODO: This class should go away eventually.
// TODO: The composition of packer + randomizer + different deserializers in a generic manner is done in the CompositeDataReader.
// TODO: Currently preserving this for backward compatibility with current configs.
CNTKTextFormatReader::CNTKTextFormatReader(const ConfigParameters& config)
{
    TextConfigHelper configHelper(config);

    try
    {
        auto corpus = make_shared<CorpusDescriptor>(true);
        if (configHelper.GetElementType() == ElementType::tfloat)
            m_deserializer = make_shared<TextParser<float>>(corpus, configHelper, true);
        else
            m_deserializer = make_shared<TextParser<double>>(corpus, configHelper, true);

        if (configHelper.ShouldKeepDataInMemory())
            m_deserializer = make_shared<ChunkCache>(m_deserializer);

        size_t window = configHelper.GetRandomizationWindow();
        if (window > 0)
        {
            // TODO: drop "verbosity", use config.traceLevel() instead. 
            int verbosity = config(L"verbosity", 0); 
            m_sequenceEnumerator = make_shared<BlockRandomizer>(verbosity, window, m_deserializer,
                                                                /*shouldPrefetch =*/ true,
                                                                /*multithreadedGetNextSequences =*/ false,
                                                                /*maxNumberOfInvalidSequences =*/ 0,
                                                                /*sampleBasedRandomizationWindow =*/ configHelper.UseSampleBasedRandomizationWindow());
        }
        else
        {
            m_sequenceEnumerator = make_shared<NoRandomizer>(m_deserializer);
        }

        if (configHelper.IsInFrameMode()) 
        {
            m_packer = std::make_shared<FramePacker>(
                m_sequenceEnumerator,
                ReaderBase::GetStreamDescriptions());
        }
        else
        {
            m_packer = std::make_shared<SequencePacker>(
                m_sequenceEnumerator,
                ReaderBase::GetStreamDescriptions());
        }
    }
    catch (const std::runtime_error& e)
    {
        RuntimeError("CNTKTextFormatReader: While reading '%ls': %s", configHelper.GetFilePath().c_str(), e.what());
    }
}

} } }
