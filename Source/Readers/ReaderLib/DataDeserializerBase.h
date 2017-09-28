//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"

namespace CNTK {

class Index;

// Base class for data deserializers.
// Has a default implementation for a subset of methods.
class DataDeserializerBase : public DataDeserializer
{
public:
    DataDeserializerBase(bool primary) : m_primary(primary)
    {}

    virtual bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo& result) override
    {
        return GetSequenceInfoByKey(primary.m_key, result);
    }

    virtual std::vector<StreamInformation> StreamInfos() override
    {
        return m_streams;
    }

protected:
    virtual bool GetSequenceInfoByKey(const SequenceKey&, SequenceInfo&)
    {
        NOT_IMPLEMENTED;
    }

    bool GetSequenceInfoByKey(const Index& index, const SequenceKey& key, SequenceInfo& r);

    // Streams this data deserializer can produce.
    std::vector<StreamInformation> m_streams;

    // Flag, indicating if the deserializer is primary.
    const bool m_primary;

private:
    DISABLE_COPY_AND_MOVE(DataDeserializerBase);
};

}
