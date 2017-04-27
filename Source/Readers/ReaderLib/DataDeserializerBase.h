//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Base class for data deserializers.
// Has a default implementation for a subset of methods.
class DataDeserializerBase : public IDataDeserializer
{
public:
    DataDeserializerBase(bool primary) : m_primary(primary)
    {}

    virtual bool GetSequenceDescription(const SequenceDescription& primary, SequenceDescription& result) override
    {
        return GetSequenceDescriptionByKey(primary.m_key, result);
    }

    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_streams;
    }

protected:
    virtual bool GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&)
    {
        NOT_IMPLEMENTED;
    }

    // Streams this data deserializer can produce.
    std::vector<StreamDescriptionPtr> m_streams;

    // Flag, indicating if the deserializer is primary.
    const bool m_primary;

private:
    DISABLE_COPY_AND_MOVE(DataDeserializerBase);
};

}}}
