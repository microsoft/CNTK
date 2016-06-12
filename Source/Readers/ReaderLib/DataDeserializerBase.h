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
    DataDeserializerBase()
    {}

    virtual bool GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&) override
    {
        NOT_IMPLEMENTED;
    }

    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_streams;
    }

protected:
    // Streams this data deserializer can produce.
    std::vector<StreamDescriptionPtr> m_streams;

private:
    DISABLE_COPY_AND_MOVE(DataDeserializerBase);
};

}}}
