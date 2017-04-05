//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "Config.h"
#include "ReaderShim.h"
#include "HTKMLFReader.h"
#include "HeapMemoryProvider.h"
#include "HTKDeserializer.h"
#include "MLFDeserializer.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// For old config, we have to emulate the same behavior as the old reader.
template<class ElemType>
class HTKMLFReaderShim : public ReaderShim<ElemType>
{
public:
    explicit HTKMLFReaderShim(ReaderFactory f) : ReaderShim<ElemType>(f) {}

    bool IsLegacyReader() const override
    {
        return true;
    }
};


// Factory methods for the reader.
// TODO: Must be removed when SGD is moved to an untyped matrix.
auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<HTKMLFReader>(parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new HTKMLFReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new HTKMLFReaderShim<double>(factory);
}

// TODO: Not safe from the ABI perspective. Will be uglified to make the interface ABI.
extern "C" DATAREADER_API bool CreateDeserializer(IDataDeserializer** deserializer, const std::wstring& type, const ConfigParameters& deserializerConfig, CorpusDescriptorPtr corpus,  bool primary)
{
    if (type == L"HTKFeatureDeserializer")
    {
        *deserializer = new HTKDeserializer(corpus, deserializerConfig, primary);
    }
    else if (type == L"HTKMLFDeserializer")
    {
        *deserializer = new MLFDeserializer(corpus, deserializerConfig, primary);
    }
    else
    {
        // Unknown type.
        return false;
    }

    // Deserializer created.
    return true;
}

}}}
