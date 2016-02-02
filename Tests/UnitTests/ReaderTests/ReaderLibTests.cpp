//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#include "BlockRandomizer.h"
#include "DataDeserializer.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(ReaderLibTests)

class MockDeserializer : public DataDeserializer
{
private:
    SequenceDescriptions m_sequenceDescriptions;

public:
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        std::vector<StreamDescriptionPtr> result;
        return result;
    }

    const SequenceDescriptions& GetSequenceDescriptions() const override
    {
        return m_sequenceDescriptions;
    }

    void StartEpoch(const EpochConfiguration& config) override
    {
        UNREFERENCED_PARAMETER(config);
    }

    std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override
    {
        UNREFERENCED_PARAMETER(ids);
        return std::vector<std::vector<SequenceDataPtr>>();
    }

    void RequireChunk(size_t chunkIndex) override
    {
        UNREFERENCED_PARAMETER(chunkIndex);
    }

    void ReleaseChunk(size_t chunkIndex) override
    {
        UNREFERENCED_PARAMETER(chunkIndex);
    }
};

BOOST_AUTO_TEST_CASE(BlockRandomizerInstantiate)
{
    auto mockDeserializer = std::make_shared<MockDeserializer>();

    auto randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer);
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
