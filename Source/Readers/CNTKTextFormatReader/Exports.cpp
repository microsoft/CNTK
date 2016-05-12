//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "ReaderShim.h"
#include "CNTKTextFormatReader.h"
#include "HeapMemoryProvider.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: Memory provider should be injected by SGD.

auto factory = [](const ConfigParameters& parameters) -> ReaderPtr
{
    return std::make_shared<CNTKTextFormatReader>(std::make_shared<HeapMemoryProvider>(), parameters);
};

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new ReaderShim<float>(factory);
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new ReaderShim<double>(factory);
}

// TODO: Not safe from the ABI perspective. Will be uglified to make the interface ABI.
// A factory method for creating image deserializers.
extern "C" DATAREADER_API bool CreateDeserializer(IDataDeserializer** deserializer, const std::wstring& type, const ConfigParameters& deserializerConfig, CorpusDescriptorPtr corpus, bool)
{
    string precision = deserializerConfig.Find("precision", "float");
    if (type == L"CNTKTextFormatDeserializer")
    {
        if (precision == "float")
            *deserializer = new TextParser<float>(corpus, TextConfigHelper(deserializerConfig));
        else // Currently assume double, TODO: should change when support more types.
            *deserializer = new TextParser<double>(corpus, TextConfigHelper(deserializerConfig));
    }
    else
        // Unknown type.
        return false;

    // Deserializer created.
    return true;
}


}}}
